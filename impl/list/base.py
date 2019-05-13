import util
from . import parser as list_parser
from . import store as list_store


def get_parsed_listpages() -> list:
    global __PARSED_LISTPAGES__
    if '__PARSED_LISTPAGES__' not in globals():
        __PARSED_LISTPAGES__ = util.load_or_create_cache('dbpedia_listpage_parsed', _compute_parsed_listpages)

    return __PARSED_LISTPAGES__


def _compute_parsed_listpages() -> list:
    listpages = list_store.get_listpages()
    listpage_markup = {lp: list_store.get_listpage_markup(lp) for lp in listpages}

    return [list_parser.parse_entries(listpage_markup[lp]) for lp in listpages]
