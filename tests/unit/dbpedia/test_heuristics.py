import impl.dbpedia.heuristics as dbp_heur
from impl.dbpedia.util import NAMESPACE_DBP_ONTOLOGY as dbo


def test_disjoint_types():
    # Person
    assert f'{dbo}Activity' in dbp_heur.get_disjoint_types(f'{dbo}Person')
    assert f'{dbo}Event' in dbp_heur.get_disjoint_types(f'{dbo}Person')
    assert f'{dbo}Place' in dbp_heur.get_disjoint_types(f'{dbo}Person')
    assert f'{dbo}Organisation' in dbp_heur.get_disjoint_types(f'{dbo}Person')
    assert f'{dbo}Work' in dbp_heur.get_disjoint_types(f'{dbo}Person')
    assert f'{dbo}Artist' not in dbp_heur.get_disjoint_types(f'{dbo}Person')
    assert f'{dbo}Agent' not in dbp_heur.get_disjoint_types(f'{dbo}Person')
    # Place
    assert f'{dbo}Award' in dbp_heur.get_disjoint_types(f'{dbo}Place')
    assert f'{dbo}EthnicGroup' in dbp_heur.get_disjoint_types(f'{dbo}Place')
    assert f'{dbo}Food' in dbp_heur.get_disjoint_types(f'{dbo}Place')
    assert f'{dbo}MeanOfTransportation' in dbp_heur.get_disjoint_types(f'{dbo}Place')
    assert f'{dbo}Work' in dbp_heur.get_disjoint_types(f'{dbo}Place')
    assert f'{dbo}Organisation' not in dbp_heur.get_disjoint_types(f'{dbo}Place')
    assert f'{dbo}Company' not in dbp_heur.get_disjoint_types(f'{dbo}Place')
    assert f'{dbo}University' not in dbp_heur.get_disjoint_types(f'{dbo}Place')
    assert f'{dbo}School' not in dbp_heur.get_disjoint_types(f'{dbo}Place')
    # Organisation
    assert f'{dbo}Activity' in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Event' in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Award' in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Device' in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Infrastructure' in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Work' in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}UnitOfWork' in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Species' in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Hospital' not in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Casino' not in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Museum' not in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    assert f'{dbo}Restaurant' not in dbp_heur.get_disjoint_types(f'{dbo}Organisation')
    # Event
    assert f'{dbo}Activity' in dbp_heur.get_disjoint_types(f'{dbo}Event')
    assert f'{dbo}Agent' in dbp_heur.get_disjoint_types(f'{dbo}Event')
    assert f'{dbo}Device' in dbp_heur.get_disjoint_types(f'{dbo}Event')
    assert f'{dbo}Food' in dbp_heur.get_disjoint_types(f'{dbo}Event')
    assert f'{dbo}MeanOfTransportation' in dbp_heur.get_disjoint_types(f'{dbo}Event')
    assert f'{dbo}Species' in dbp_heur.get_disjoint_types(f'{dbo}Event')
    assert f'{dbo}Work' in dbp_heur.get_disjoint_types(f'{dbo}Event')
    # Species
    assert f'{dbo}Agent' in dbp_heur.get_disjoint_types(f'{dbo}Species')
    assert f'{dbo}Event' in dbp_heur.get_disjoint_types(f'{dbo}Species')
    assert f'{dbo}Activity' in dbp_heur.get_disjoint_types(f'{dbo}Species')
    assert f'{dbo}Disease' in dbp_heur.get_disjoint_types(f'{dbo}Species')
    assert f'{dbo}Place' in dbp_heur.get_disjoint_types(f'{dbo}Species')
    assert f'{dbo}Work' in dbp_heur.get_disjoint_types(f'{dbo}Species')
    assert f'{dbo}Award' in dbp_heur.get_disjoint_types(f'{dbo}Species')
    # Work
    assert f'{dbo}Agent' in dbp_heur.get_disjoint_types(f'{dbo}Work')
    assert f'{dbo}Activity' in dbp_heur.get_disjoint_types(f'{dbo}Work')
    assert f'{dbo}AnatomicalStructure' in dbp_heur.get_disjoint_types(f'{dbo}Work')
    assert f'{dbo}Food' in dbp_heur.get_disjoint_types(f'{dbo}Work')
    assert f'{dbo}TimePeriod' in dbp_heur.get_disjoint_types(f'{dbo}Work')
    # Album
    assert f'{dbo}Agent' in dbp_heur.get_disjoint_types(f'{dbo}Album')
    assert f'{dbo}Artwork' in dbp_heur.get_disjoint_types(f'{dbo}Album')
    assert f'{dbo}Event' in dbp_heur.get_disjoint_types(f'{dbo}Album')
    assert f'{dbo}Language' in dbp_heur.get_disjoint_types(f'{dbo}Album')
    assert f'{dbo}Musical' in dbp_heur.get_disjoint_types(f'{dbo}Album')
    assert f'{dbo}Single' in dbp_heur.get_disjoint_types(f'{dbo}Album')
    # Film
    assert f'{dbo}WrittenWork' in dbp_heur.get_disjoint_types(f'{dbo}Film')
    assert f'{dbo}Artwork' in dbp_heur.get_disjoint_types(f'{dbo}Film')
    assert f'{dbo}Agent' in dbp_heur.get_disjoint_types(f'{dbo}Film')
    assert f'{dbo}Place' in dbp_heur.get_disjoint_types(f'{dbo}Film')
