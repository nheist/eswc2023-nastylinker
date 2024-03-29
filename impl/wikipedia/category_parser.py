from typing import Tuple, Optional, Dict, Set
import utils
from utils import get_logger
import impl.util.string as str_util
from impl.util.rdf import Namespace
from collections import defaultdict
import wikitextparser as wtp
from wikitextparser import WikiText, Template
import re
from tqdm import tqdm
import multiprocessing as mp


def _extract_parent_categories_from_markup(categories_and_templates_markup: Tuple[Dict[str, str], Dict[str, str]]) -> Dict[str, Set[str]]:
    get_logger().info('Extracting parent categories from category markup..')
    categories_markup, templates_markup = categories_and_templates_markup
    template_definitions = _prepare_template_definitions(templates_markup)

    data = [(cat, markup, template_definitions) for cat, markup in categories_markup.items()]
    with mp.Pool(processes=utils.get_config('max_cpus')) as pool:
        parent_categories = {cat: parents for cat, parents in tqdm(pool.imap_unordered(_extract_parents_for_category, data, chunksize=10000), total=len(data), desc='wikipedia/category_parser: Extracting parent categories')}
    return parent_categories


def _extract_parents_for_category(data: tuple) -> tuple:
    cat, markup, template_definitions = data
    content, visited_templates = _replace_templates_in_category(wtp.parse(markup), template_definitions)
    parent_names = {link.target[len(Namespace.PREFIX_CATEGORY.value):] for link in content.wikilinks if link.target.startswith(Namespace.PREFIX_CATEGORY.value)}
    parent_names = map(str_util.regularize_spaces, parent_names)
    parent_names = map(str_util.capitalize, parent_names)  # make sure that first letter of category is uppercase
    parent_uris = {Namespace.DBP_CATEGORY.value + name.replace(' ', '_') for name in parent_names}
    # use markers in markup or used templates to find parent categories (as we can't resolve macros in markup)
    if '__HIDDENCAT__' in content.string or 'maintenance category' in visited_templates:
        parent_uris.add(f'{Namespace.DBP_CATEGORY.value}Hidden_categories')
    if 'category disambiguation' in visited_templates or 'category ambiguous' in visited_templates:
        parent_uris.add(f'{Namespace.DBP_CATEGORY.value}Disambiguation_categories')
    if 'category redirect' in visited_templates or 'cat redirect' in visited_templates:
        parent_uris.add(f'{Namespace.DBP_CATEGORY.value}Wikipedia_soft_redirected_categories')
    return cat, parent_uris


# PREPARE & APPLY TEMPLATES


def _prepare_template_definitions(templates_markup: dict) -> dict:
    template_definitions = defaultdict(str)
    # extract parts of the template that will be placed on the page the template is applied to
    for name, content in templates_markup.items():
        name = str_util.capitalize(name[len(Namespace.DBP_TEMPLATE.value):].replace('_', ' '))
        content = re.sub(r'</?includeonly>', '', content)  # remove includeonly-tags
        content = re.sub(r'<noinclude>(.|\n)*?</noinclude>', '', content)  # remove content in noinclude-tags
        content = _filter_for_onlyinclude(content)
        template_definitions[name] = content
    # handle redirects in templates
    for name in set(template_definitions):
        content = template_definitions[name]
        if content.startswith('#REDIRECT'):
            parse = wtp.parse(content[len('#REDIRECT '):])
            if not parse.wikilinks or not parse.wikilinks[0].target.startswith(Namespace.PREFIX_TEMPLATE.value):
                template_definitions[name] = ''
            else:
                target = parse.wikilinks[0].target[len(Namespace.PREFIX_TEMPLATE.value):]
                template_definitions[name] = template_definitions[target]
    return template_definitions


ONLYINCLUDE_START_TAG = '<onlyinclude>'
ONLYINCLUDE_END_TAG = '</onlyinclude>'
def _filter_for_onlyinclude(text: str) -> str:
    if ONLYINCLUDE_START_TAG not in text:
        return text
    if ONLYINCLUDE_END_TAG not in text:
        return ''  # start tag without end tag
    onlyinclude_start = text.index(ONLYINCLUDE_START_TAG) + len(ONLYINCLUDE_START_TAG)
    onlyinclude_end = text.index(ONLYINCLUDE_END_TAG)
    return text[onlyinclude_start:onlyinclude_end]


def _replace_templates_in_category(category_content: WikiText, template_definitions: dict) -> Tuple[WikiText, set]:
    all_visited_templates = set()
    for template in category_content.templates:
        template_content, visited_templates = _get_template_content(template, template_definitions, set())
        if template_content is None:
            continue
        category_content[slice(*template.span)] = template_content
        all_visited_templates.update({t.lower() for t in visited_templates})
    return category_content, all_visited_templates


def _get_template_content(template: Template, template_definitions: dict, visited_templates: set) -> Tuple[Optional[str], set]:
    if not template or not template.string.startswith('{{'):
        return None, visited_templates
    try:
        name = template.normal_name(capitalize=True)
    except IndexError:
        return '', visited_templates
    if name.lower() in visited_templates:
        return '', visited_templates
    visited_templates.add(name.lower())

    content = wtp.parse(template_definitions[name])
    content = _apply_parameters(content, _get_template_arguments(template))
    for it in content.templates:
        it_content, _ = _get_template_content(it, template_definitions, visited_templates)
        if it_content is not None:
            content[slice(*it.span)] = it_content
    return content.string, visited_templates


def _get_template_arguments(template: Template) -> dict:
    args = {}
    for arg in template.arguments:
        args[arg.name.strip('\r\n\t ')] = arg.value
    return args


def _apply_parameters(content: WikiText, arguments: dict) -> WikiText:
    for p in content.parameters:
        if not p:
            continue
        param_value = arguments[p.name] if p.name in arguments else _apply_parameters(wtp.parse(p.default or ''), arguments).string
        content[slice(*p.span)] = param_value
    return content
