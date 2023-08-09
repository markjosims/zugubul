from zugubul.utils import eaf_to_file_safe
from pympi import Elan
import os

def test_eaf_to_file_safe():
    eaf_fp = r'C:\projects\zugubul\tests\eafs\test_tira1_gold.eaf'
    bak_fp = r'C:\projects\zugubul\tests\eafs\test_tira1_gold.bak'
    eaf = Elan.Eaf(eaf_fp)
    bak = Elan.Eaf(bak_fp)

    eaf_to_file_safe(eaf, eaf_fp)

    assert os.path.isfile(eaf_fp)   
    assert os.path.isfile(bak_fp)   