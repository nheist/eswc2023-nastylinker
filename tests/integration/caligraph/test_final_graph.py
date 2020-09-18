from pytest_check import check_func
import impl.caligraph.base as cali_base
from impl.caligraph.util import NAMESPACE_CLG_ONTOLOGY as CLGO
from impl.dbpedia.util import NAMESPACE_DBP_ONTOLOGY as DBO
from impl.dbpedia.util import NAMESPACE_DBP_RESOURCE as DBR


def test_classes():
    _is_not_in_graph(f'{CLGO}Demographic')  # TODO: Fix! (e.g. with better non-set category identification)


def test_class_hierarchy():
    _is_parent_of(f'{CLGO}Air_force', f'{CLGO}Disbanded_air_force')
    _is_no_parent_of(f'{CLGO}Air_force', f'{CLGO}Air_force_personnel')  # TODO: Fixed?

    _is_parent_of(f'{CLGO}Bodybuilder', f'{CLGO}Fitness_or_figure_competitor')  # TODO: Fix!
    _is_no_parent_of(f'{CLGO}Fitness_or_figure_competition', f'{CLGO}Fitness_or_figure_competitor')  # TODO: Fixed?

    _is_parent_of(f'{CLGO}Bodybuilding_competition', f'{CLGO}Female_bodybuilding_competition')  # TODO: Fix!
    _is_no_parent_of(f'{CLGO}Person', f'{CLGO}Female_bodybuilding_competition')  # TODO: Fixed?

    _is_parent_of(f'{CLGO}University_or_college_person', f'{CLGO}Alumni')
    _is_no_parent_of(f'{CLGO}Honduran_person', f'{CLGO}Alumni')  # TODO: Fix!
    _is_no_parent_of(f'{CLGO}Papua_New_Guinean_person', f'{CLGO}Alumni')  # TODO: Fix!

    _is_no_parent_of(f'{CLGO}Japanese_sportsperson', f'{CLGO}Association_football_person')  # TODO: Fix!
    _is_parent_of(f'{CLGO}Football_person', f'{CLGO}Association_football_person')

    _is_ancestor_of(f'{CLGO}Person', f'{CLGO}Sportswoman')  # TODO: Fixed?


@check_func
def _is_parent_of(parent: str, child: str):
    G = cali_base.get_axiom_graph()
    assert child in G.children(parent)


@check_func
def _is_ancestor_of(ancestor: str, child: str):
    G = cali_base.get_axiom_graph()
    assert ancestor in G.ancestors(child)


@check_func
def _is_no_parent_of(parent: str, child: str):
    G = cali_base.get_axiom_graph()
    assert child not in G.children(parent)


def test_node_parts():
    _is_part_of(f'{DBO}Organisation', f'{CLGO}Organization')  # TODO: Fixed?


def test_singularization():
    _is_part_of(f'{DBR}Category:Engineering_societies_by_country', f'{CLGO}Engineering_society')  # TODO: Fixed?
    _is_part_of(f'{DBR}List_of_engineering_societies', f'{CLGO}Engineering_society')

    _is_part_of(f'{DBR}List_of_sportswomen', f'{CLGO}Sportswoman')  # TODO: Fixed?

    _is_no_part_of(f'{DBR}List_of_caves', f'{CLGO}Cafe')  # TODO: Fixed?


def test_by_phrase_removal():
    _is_part_of(f'{DBR}List_of_countries_by_national_capital_and_largest_cities', f'{CLGO}Country')  # TODO: Fix!
    _is_part_of(f'{DBR}List_of_countries_by_vehicles_per_capita', f'{CLGO}Country')  # TODO: Fix!
    _is_part_of(f'{DBR}List_of_countries_by_Nobel_laureates_per_capita', f'{CLGO}Country')  # TODO: Fix!
    _is_part_of(f'{DBR}List_of_countries_by_number_of_households', f'{CLGO}Country')  # TODO: Fix!
    _is_part_of(f'{DBR}List_of_countries_by_health_expenditure_covered_by_government', f'{CLGO}Country')  # TODO: Fixed?
    _is_part_of(f'{DBR}Category:Lists_of_countries_by_per_capita_values', f'{CLGO}Country')  # TODO: Fix!
    _is_part_of(f'{DBR}Category:Lists_of_countries_by_GDP_per_capita', f'{CLGO}Country')  # TODO: Fixed?
    _is_part_of(f'{DBR}Category:Vehicles_by_brand_controlled_by_Volkswagen_Group', f'{CLGO}Vehicle')  # TODO: Fixed?
    _is_part_of(f'{DBR}Category:African_films_by_genre_by_country', f'{CLGO}African_film')  # TODO: Fixed?

    _is_in_graph(f'{CLGO}Work_by_Roy_Lichtenstein')
    _is_in_graph(f'{CLGO}Film_produced_by_Harry_Saltzman')
    _is_in_graph(f'{CLGO}20th-century_execution_by_Sweden')
    _is_in_graph(f'{CLGO}United_States_Article_I_federal_judge_appointed_by_Jimmy_Carter')
    _is_in_graph(f'{CLGO}Bundesliga_club_eliminated_from_the_DFB-Pokal_by_amateur_sides')
    _is_in_graph(f'{CLGO}Opera_by_Krenek')
    _is_in_graph(f'{CLGO}Work_by_Presidents_of_the_United_States')
    _is_in_graph(f'{CLGO}Electronica_album_by_South_African_artists')
    _is_in_graph(f'{CLGO}Person_killed_by_live_burial_by_Nazi_Germany')
    _is_in_graph(f'{CLGO}Killing_by_law_enforcement_officers_by_countries')
    _is_in_graph(f'{CLGO}Pop_album_by_Scottish_artists')
    _is_in_graph(f'{CLGO}Song_recorded_by_ABBA')
    _is_in_graph(f'{CLGO}Program_broadcast_by_France_2')
    _is_in_graph(f'{CLGO}International_cycle_race_hosted_by_the_United_Kingdom')


@check_func
def _is_in_graph(node: str):
    G = cali_base.get_axiom_graph()
    assert node in G.nodes


@check_func
def _is_not_in_graph(node: str):
    G = cali_base.get_axiom_graph()
    assert node not in G.nodes


@check_func
def _is_part_of(part: str, node: str):
    G = cali_base.get_axiom_graph()
    assert part in G.get_parts(node)


@check_func
def _is_no_part_of(part: str, node: str):
    G = cali_base.get_axiom_graph()
    assert part not in G.get_parts(node)
