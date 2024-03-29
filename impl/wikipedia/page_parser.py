"""Functionality for parsing Wikipedia pages from WikiText."""

from typing import Tuple, Optional, Dict, Set, List, Iterable
from collections import defaultdict, namedtuple, Counter
import re
import signal
import traceback
from tqdm import tqdm
import multiprocessing as mp
import wikitextparser as wtp
from wikitextparser import WikiText
import utils
from impl.util.nlp import EntityTypeLabel
from impl.util.transformer import EntityIndex
from impl.util.spacy import listing_parser, get_tokens_and_whitespaces_from_text
from impl.dbpedia.resource import DbpResource, DbpFile, DbpResourceStore
from impl.dbpedia.util import is_entity_name
from . import wikimarkup_parser as wmp


ListingId = namedtuple('ListingId', ['page_idx', 'listing_idx'])
MentionId = namedtuple('MentionId', ['page_idx', 'listing_idx', 'item_idx'])


class WikiSubjectEntity:
    def __init__(self, entity_idx: int, label: str, entity_type: EntityTypeLabel):
        self.entity_idx = entity_idx
        self.label = label
        self.entity_type = entity_type


class WikiMention:
    def __init__(self, name: Optional[str], label: str, start: int, end: int):
        self.name = name
        self.entity_idx = None  # set after parsing
        self.label = label
        self.start = start
        self.end = end


class WikiListingItem:
    def __init__(self, idx: int):
        self.idx = idx
        self.subject_entity = None

    def get_mentions(self) -> List[WikiMention]:
        raise NotImplementedError()


class WikiEnumEntry(WikiListingItem):
    def __init__(self, idx: int, tokens: List[str], whitespaces: List[str], mentions: List[WikiMention], depth: int, is_leaf: bool):
        super().__init__(idx)
        self.tokens = tokens
        self.whitespaces = whitespaces
        self.mentions = mentions
        self.depth = depth
        self.is_leaf = is_leaf

    def get_mentions(self) -> List[WikiMention]:
        return self.mentions


class WikiTableRow(WikiListingItem):
    def __init__(self, idx: int, tokens: List[List[str]], whitespaces: List[List[str]], mentions: List[List[WikiMention]]):
        super().__init__(idx)
        self.tokens = tokens
        self.whitespaces = whitespaces
        self.mentions = mentions

    def get_mentions(self) -> List[WikiMention]:
        return [m for cell_mentions in self.mentions for m in cell_mentions]


class WikiSection:
    def __init__(self, section_data, mention_resource_mapping: dict):
        raw_title = section_data.title.strip() if section_data.title and section_data.title.strip() else 'Main'
        self.title = wmp.wikitext_to_plaintext(raw_title)
        self.tokens, self.whitespaces = get_tokens_and_whitespaces_from_text(self.title)
        self.level = section_data.level
        self.resource_name = wmp.get_first_wikilink_resource(raw_title) or _find_first_entity_with_mention_resource_mapping(self.tokens, mention_resource_mapping)
        self.entity_idx = None  # set after parsing

    def is_top_section(self) -> bool:
        return self.level <= 2

    def is_meta_section(self) -> bool:
        return self.title.lower() in {
            'see also', 'external links', 'references', 'notes', 'sources', 'external sources', 'general sources',
            'bibliography', 'notes and references', 'citations', 'references and footnotes', 'references and links',
            'maps', 'further reading'
        }


class WikiListing:
    def __init__(self, idx: int, topsection: WikiSection, section: WikiSection, items: List[WikiListingItem]):
        self.idx = idx
        self.topsection = topsection
        self.section = section
        self.items = {item.idx: item for item in items}
        self.page_idx = None

    def get_id(self) -> ListingId:
        return ListingId(self.page_idx, self.idx)

    def get_sections(self) -> Set[WikiSection]:
        return {self.topsection, self.section}

    @classmethod
    def get_type(cls) -> str:
        raise NotImplementedError()

    def get_items(self, has_subject_entity: bool = False, has_known_entity: bool = False) -> List[WikiListingItem]:
        if not has_subject_entity:
            return list(self.items.values())
        items = [i for i in self.items.values() if i.subject_entity is not None]
        if not has_known_entity:
            return items
        return [i for i in items if i.subject_entity.entity_idx != EntityIndex.NEW_ENTITY]

    def get_mentioned_entities(self) -> Set[DbpResource]:
        dbr = DbpResourceStore.instance()
        entity_indices = {mention.entity_idx for item in self.get_items() for mention in item.get_mentions()}
        return {dbr.get_resource_by_idx(idx) for idx in entity_indices if idx != EntityIndex.NEW_ENTITY}

    def get_subject_entities(self) -> List[WikiSubjectEntity]:
        return [item.subject_entity for item in self.get_items(has_subject_entity=True)]

    def has_subject_entities(self) -> bool:
        return len(self.get_subject_entities()) > 0

    def has_main_subject_entity(self) -> bool:
        """Return True, if there is a known subject entity that appears in at least 3/4 of all items."""
        subject_entities = self.get_subject_entities()
        named_se_counter = Counter([se.entity_idx for se in subject_entities if se.entity_idx != EntityIndex.NEW_ENTITY])
        if not named_se_counter:
            return False
        return named_se_counter.most_common(1)[0][1] / len(subject_entities) >= .75


class WikiEnum(WikiListing):
    @classmethod
    def get_type(cls) -> str:
        return cls.__name__


class WikiTable(WikiListing):
    def __init__(self, idx: int, topsection: WikiSection, section: WikiSection, items: List[WikiTableRow], header: WikiTableRow):
        super().__init__(idx, topsection, section, items)
        self.header = header

    @classmethod
    def get_type(cls) -> str:
        return cls.__name__


class WikiPage:
    def __init__(self, name: str, listings: List[WikiListing]):
        self.name = name
        self.idx = None  # set after parsing
        self.listings = {listing.idx: listing for listing in listings}

    @property
    def resource(self) -> DbpResource:
        return DbpResourceStore.instance().get_resource_by_idx(self.idx)

    def get_sections(self) -> Set[WikiSection]:
        return {section for listing in self.get_listings() for section in listing.get_sections()}

    def get_listings(self) -> Iterable[WikiListing]:
        return self.listings.values()

    def has_subject_entities(self) -> bool:
        return any(listing.has_subject_entities() for listing in self.get_listings())

    def get_subject_entities(self) -> List[WikiSubjectEntity]:
        return [se for listing in self.get_listings() for se in listing.get_subject_entities()]


LISTING_INDICATORS = ('*', '#', '{|')
VALID_ENUM_PATTERNS = (r'\#', r'\*')


def _parse_pages(pages_markup: Dict[str, str]) -> List[WikiPage]:
    # prepare and filter markup
    markup_iterator = tqdm(pages_markup.items(), total=len(pages_markup), desc='wikipedia/page_parser: Preparing pages')
    with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
        pages_markup = {res: markup for res, markup in pool.imap_unordered(_prepare_page_markup, markup_iterator, chunksize=5000) if res and markup}
    # process markup
    wikipedia_pages = []
    with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
        markup_iterator = tqdm(pages_markup.items(), total=len(pages_markup), desc=f'wikipedia/page_parser: Parsing pages')
        for res, wp in pool.imap_unordered(_parse_page_with_timeout, markup_iterator, chunksize=5000):
            if wp is not None and wp.get_listings():
                wikipedia_pages.append(wp)
            pages_markup[res] = ''  # discard markup after parsing to free memory
    # finalize pages
    cleaned_wikipedia_pages = []
    dbr = DbpResourceStore.instance()
    for page in tqdm(wikipedia_pages, desc='wikipedia/page_parser: Finalizing pages'):
        # check page validity
        if not dbr.has_resource_with_name(page.name):
            continue
        res = dbr.get_resource_by_name(page.name)
        if isinstance(res, DbpFile) or res.is_meta:
            continue
        # set page index
        page.idx = res.idx
        for listing in page.get_listings():
            listing.page_idx = page.idx
        # update section entities
        for section in page.get_sections():
            section.entity_idx = wmp.get_resource_idx_for_resource_name(section.resource_name)
        # update mention entities and discard invalid mentions
        for listing in page.get_listings():
            for item in listing.get_items():
                if isinstance(item, WikiEnumEntry):
                    item.mentions = _init_mention_entities(item.mentions)
                elif isinstance(item, WikiTableRow):
                    item.mentions = [_init_mention_entities(cell_mentions) for cell_mentions in item.mentions]
        cleaned_wikipedia_pages.append(page)
    return cleaned_wikipedia_pages


def _prepare_page_markup(resource_and_markup: Tuple[str, str]) -> Tuple[Optional[str], str]:
    resource, page_markup = resource_and_markup
    if not any(indicator in page_markup for indicator in LISTING_INDICATORS):
        return None, page_markup  # early return if page contains no listings at all
    # prepare markup for parsing
    page_markup = page_markup.replace('&nbsp;', ' ')  # replace html whitespaces
    page_markup = page_markup.replace('<br />', ' ')  # replace html line breaks
    page_markup = page_markup.replace('<br/>', ' ')
    page_markup = page_markup.replace('<br>', ' ')
    page_markup = re.sub(r'<ref>.*?</ref>', '', page_markup)  # remove ref markers
    page_markup = re.sub(r'<ref[^>]*?/>', '', page_markup)
    page_markup = re.sub(r"'{2,}", '', page_markup)  # remove bold and italic markers
    try:
        # early return if page is not useful
        wiki_text = wtp.parse(page_markup)
        if not _is_page_useful(wiki_text):
            return None, page_markup
        # clean and expand markup
        cleaned_wiki_text = _convert_special_enums(wiki_text)
        cleaned_wiki_text = _remove_enums_within_tables(cleaned_wiki_text)
        if not _is_page_useful(cleaned_wiki_text):
            return None, page_markup
    except Exception as e:
        if type(e) == KeyboardInterrupt:
            raise e
        utils.get_logger().error(f'Failed to prepare page {resource}: {traceback.format_exc()}')
        return None, page_markup
    return resource, cleaned_wiki_text.string


def _is_page_useful(wiki_text: WikiText) -> bool:
    # ignore pages without any lists and pages with very small lists (e.g. redirect pages have a list of length 1)
    valid_list_count = len([list for list in wiki_text.get_lists(VALID_ENUM_PATTERNS) if len(list.items) >= 3])
    valid_table_count = len([table for table in wiki_text.get_tables() if len(table.data()) >= 3])
    return valid_list_count + valid_table_count > 0


def _convert_special_enums(wiki_text: WikiText) -> WikiText:
    """Convert special templates used as enumerations from the text."""
    # convert enumeration templates
    enum_templates = [t for t in wiki_text.templates if t.name == 'columns-list']
    if enum_templates:
        result = wiki_text.string
        for et in enum_templates:
            actual_list = et.get_arg('1')
            result = result.replace(et.string, actual_list.string[1:] if actual_list else '')
        return wtp.parse(result)
    return wiki_text


def _remove_enums_within_tables(wiki_text: WikiText) -> WikiText:
    """Remove any enumeration markup that is contained within a table."""
    something_changed = False
    for t in wiki_text.tables:
        for row in t.cells():
            for cell in row:
                if cell:
                    for lst in cell.get_lists(VALID_ENUM_PATTERNS):
                        lst.convert('')
                        something_changed = True
    return wtp.parse(wiki_text.string) if something_changed else wiki_text


def _parse_page_with_timeout(resource_and_markup: Tuple[str, str]) -> Tuple[str, Optional[WikiPage]]:
    """Return the parsed wikipedia page (with empty content, if parsing has timed out)"""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(60)  # timeout of one minute per page

    resource, page_markup = resource_and_markup
    try:
        wp = _parse_page(resource, page_markup)
        signal.alarm(0)  # reset alarm as parsing was successful
        return resource, wp
    except Exception as e:
        if type(e) == KeyboardInterrupt:
            raise e
        utils.get_logger().error(f'Failed to parse page {resource}: {traceback.format_exc()}')
        return resource, None


def _parse_page(resource: str, page_markup: str) -> Optional[WikiPage]:
    wiki_text = wtp.parse(page_markup)
    mention_resource_mapping = _create_mention_resource_mapping(wiki_text, resource)
    return WikiPage(resource, _extract_listings(wiki_text, mention_resource_mapping))


def _create_mention_resource_mapping(wiki_text: WikiText, resource: str) -> dict:
    mention_to_resource = {wmp.get_label_for_wikilink(wl): wmp.get_resource_name_for_wikilink(wl) for wl in wiki_text.wikilinks}
    mention_to_resource[resource] = resource  # consider mentions of page resource itself as well
    mention_to_resource = {mention: res for mention, res in mention_to_resource.items() if mention and res}
    # tokenize every mention and index by their first word for easy retrieval
    mention_token_entities = defaultdict(list)
    for mention, res in mention_to_resource.items():
        mention_tokens, _ = get_tokens_and_whitespaces_from_text(mention)
        if not mention_tokens[0]:
            continue
        mention_token_entities[mention_tokens[0]].append((mention_tokens, mention, res))
    # sort the lists by descending length to always find the longest mention
    mention_token_entities = {idx: sorted(mte, key=lambda x: len(x[0]), reverse=True) for idx, mte in mention_token_entities.items()}
    return mention_token_entities


def _find_first_entity_with_mention_resource_mapping(tokens: List[str], mention_resource_mapping: dict) -> Optional[str]:
    for idx, token in enumerate(tokens):
        if token not in mention_resource_mapping:
            continue
        for mention_tokens, _, res in mention_resource_mapping[token]:
            mention_length = len(mention_tokens)
            if tokens[idx:idx + mention_length] == mention_tokens:
                return res
    return None


def _extract_listings(wiki_text: WikiText, mention_resource_mapping: dict) -> list:
    listings = []
    listing_idx = 0
    topsection = None
    for section_data in wiki_text.get_sections(include_subsections=False):
        section = WikiSection(section_data, mention_resource_mapping)
        topsection = section if section.is_top_section() else topsection
        if topsection.is_meta_section():
            continue  # discard meta sections
        for enum_data in section_data.get_lists(VALID_ENUM_PATTERNS):
            enum = _extract_enum(listing_idx, topsection, section, enum_data, mention_resource_mapping)
            if enum is None:
                continue
            listings.append(enum)
            listing_idx += 1
        for table_data in section_data.get_tables():
            table = _extract_table(listing_idx, topsection, section, table_data, mention_resource_mapping)
            if table is None:
                continue
            listings.append(table)
            listing_idx += 1
    return listings


def _extract_enum(enum_idx: int, topsection: WikiSection, section: WikiSection, wiki_list: wtp.WikiList, mention_resource_mapping: dict) -> Optional[WikiEnum]:
    entries = _extract_enum_entries(wiki_list, mention_resource_mapping)
    if len(entries) < 3:
        return None
    return WikiEnum(enum_idx, topsection, section, entries)


def _extract_enum_entries(wiki_list: wtp.WikiList, mention_resource_mapping: dict, item_idx: int = 0) -> List[WikiEnumEntry]:
    entries = []
    for list_item_idx, item_text in enumerate(wiki_list.items):
        tokens, whitespaces, mentions = _tokenize_wikitext(item_text, mention_resource_mapping)
        sublists = wiki_list.sublists(list_item_idx)
        entries.append(WikiEnumEntry(item_idx, tokens, whitespaces, mentions, wiki_list.level, len(sublists) == 0))
        item_idx += 1
        for sl in sublists:
            subentries = _extract_enum_entries(sl, mention_resource_mapping, item_idx=item_idx)
            entries.extend(subentries)
            item_idx += len(subentries)
    return entries


def _extract_table(table_idx: int, topsection: WikiSection, section: WikiSection, table_data: wtp.Table, mention_resource_mapping: dict) -> Optional[WikiTable]:
    header = None
    rows = []
    try:
        rows_data = table_data.data(strip=True, span=True)
        all_cell_data = table_data.cells(span=True)
        row_data_with_spans = table_data.data(strip=True, span=False)
    except Exception as e:
        if type(e) in [KeyboardInterrupt, ParsingTimeoutException]:
            raise e
        return None
    for row_idx, cells in enumerate(rows_data):
        if len(cells) < 2 or len(cells) > 100:
            return None  # ignore tables with only one or more than 100 columns (likely irrelevant or markup error)
        row_tokens, row_whitespaces, row_mentions = [], [], []
        for cell in cells:
            cell_tokens, cell_whitespaces, cell_mentions = _tokenize_wikitext(str(cell), mention_resource_mapping)
            row_tokens.append(cell_tokens)
            row_whitespaces.append(cell_whitespaces)
            row_mentions.append(cell_mentions)
        row = WikiTableRow(row_idx, row_tokens, row_whitespaces, row_mentions)
        if _is_header_row(all_cell_data, row_idx):
            header = row
            header.idx = -1
            continue
        # process data row (only use rows that are not influenced by row-/colspan)
        if not len(row_data_with_spans) > row_idx or not len(cells) == len(row_data_with_spans[row_idx]):
            continue
        rows.append(row)
    if len(rows) < 3:
        return None  # ignore tables with less than 3 data rows
    return WikiTable(table_idx, topsection, section, rows, header)


def _is_header_row(cells, row_idx: int) -> bool:
    try:
        return row_idx == 0 or any(c and c.is_header for c in cells[row_idx])
    except IndexError:
        return False  # fallback if wtp can't parse the table correctly


def _tokenize_wikitext(wiki_text: str, mention_resource_mapping: dict) -> Tuple[List[str], List[str], List[WikiMention]]:
    # preprocess markup text
    parsed_text = wtp.parse(wiki_text)
    parsed_text = _remove_file_wikilinks(parsed_text)
    parsed_text = _convert_sortname_templates(parsed_text)
    doc = listing_parser.parse(wmp.wikitext_to_plaintext(parsed_text).strip())
    tokens, whitespaces = [w.text for w in doc], [w.whitespace_ for w in doc]

    # extract wikilink-mentions with correct token positions
    mentions = []
    current_index = 0
    for w in parsed_text.wikilinks:
        text = wmp.get_label_for_wikilink(w)
        if not text:
            continue  # skip entity with empty text

        # retrieve mention position
        mention_tokens, _ = get_tokens_and_whitespaces_from_text(text)
        mention_start_idx = current_index
        while True:  # repeat as long as we find potential starting positions of the mention
            try:
                mention_start_idx = tokens.index(mention_tokens[0], mention_start_idx)
                mention_end_idx = mention_start_idx + len(mention_tokens)
                if tokens[mention_start_idx:mention_end_idx] != mention_tokens:
                    mention_start_idx += 1  # increment index to avoid running into the same mention again
                    continue  # no exact match of position and mention text; try next potential starting position
                res = wmp.get_resource_name_for_wikilink(w)
                if res is not None:
                    mentions.append(WikiMention(res, text, mention_start_idx, mention_end_idx))
                current_index = mention_end_idx
                break
            except ValueError:
                break  # no more potential starting positions for the mention
    # add additional mentions that may be labeled somewhere else on the page (mention expansion)
    tokens_with_mentions = set()
    for mention in mentions:
        tokens_with_mentions.update(set(range(mention.start, mention.end)))
    expanded_mentions = _find_expanded_mentions(tokens, mention_resource_mapping, tokens_with_mentions)
    mentions.extend(expanded_mentions)
    # add additional mentions from spacy listing parser (that are not overlapping with existing mentions)
    for ent in doc.ents:
        token_indices = set(range(ent.start, ent.end))
        if not token_indices.intersection(tokens_with_mentions):
            mentions.append(WikiMention(None, ent.text, ent.start, ent.end))
    mentions = list(sorted(mentions, key=lambda m: m.start))
    return tokens, whitespaces, mentions


def _find_expanded_mentions(tokens: List[str], mention_resource_mapping: dict, tokens_with_mentions: Set[int]) -> List[WikiMention]:
    # go over every token and check whether it is the start of an undetected mention
    expanded_mentions = []
    for idx, token in enumerate(tokens):
        if idx in tokens_with_mentions or token not in mention_resource_mapping:
            continue
        for mention_tokens, mention, res in mention_resource_mapping[token]:
            mention_length = len(mention_tokens)
            if tokens[idx:idx+mention_length] == mention_tokens:
                expanded_mentions.append(WikiMention(res, mention, idx, idx+mention_length))
                tokens_with_mentions.update(set(range(idx, idx+mention_length)))
                break
    return expanded_mentions


def _remove_file_wikilinks(parsed_text: wtp.WikiText) -> wtp.WikiText:
    """Remove wikilinks to files or images."""
    for wl in reversed(parsed_text.wikilinks):
        if wl.string.startswith(('[[File:', '[[Image:')):
            parsed_text[slice(*wl.span)] = ''
    return parsed_text


def _convert_sortname_templates(parsed_text: wtp.WikiText) -> wtp.WikiText:
    """Convert Sortname template (typically found in tables) into a simple wikilink.

    Documentation of Sortname template: https://en.wikipedia.org/wiki/Template:Sortname
    """
    for t in parsed_text.templates:
        if not t.string.startswith('{{'):
            continue
        if t.normal_name(capitalize=True) != 'Sortname':
            continue
        if not t.has_arg('1') or not t.has_arg('2'):
            continue
        text = (t.get_arg('1').value + ' ' + t.get_arg('2').value).strip()
        if t.has_arg('nolink'):
            result = text
        else:
            if t.has_arg('3'):
                link = t.get_arg('3').value
                result = f'[[{link}|{text}]]'
            else:
                result = f'[[{text}]]'
        parsed_text[slice(*t.span)] = result
    return parsed_text


def _init_mention_entities(mentions: List[WikiMention]) -> List[WikiMention]:
    for mention in mentions:
        ent_idx = wmp.get_resource_idx_for_resource_name(mention.name)
        if ent_idx is None:
            ent_idx = -1 if mention.name is None or is_entity_name(mention.name) else None
        mention.entity_idx = ent_idx
    return [m for m in mentions if m.entity_idx is not None]


# define functionality for parsing timeouts


class ParsingTimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ParsingTimeoutException('Parsing timeout.')
