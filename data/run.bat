python scripts\pipeline_targets_main.py --output data\output\output.targets.csv --with-orthologs  --with-isoforms  --iuphar-target dictionary\_target\_IUPHAR\_IUPHAR_target.csv  --iuphar-family dictionary\_target\_IUPHAR\_IUPHAR_family.csv --input data\input\full\target.csv  


python scripts/chembl_activities_main.py --input data\input/activity1.csv --output data/input/output.activities.csv     --column activity_chembl_id --limit 1000     --chunk-size 100     --log-level DEBUG     --log-format json

python scripts/chembl_assays_main.py --input data/input/assay1.csv --output data/input/output.assay.csv     --column assay_chembl_id      --chunk-size 10     --log-level DEBUG     --log-format json

python scripts/chembl_testitems_main.py --input data/input/testitem1.csv --output data/input/output.testitem.csv     --column molecule_chembl_id      --chunk-size 100     --log-level DEBUG     --log-format json

python scripts/pubmed_main.py --input data/input/document1.csv --output data/input/output.document.csv     --column document_chembl_id      --chunk-size 100     --log-level DEBUG     --log-format json


===============
python scripts\pubmed_main.py --output data\output\output.document_20250920.csv     --column document_chembl_id      --chunk-size 100     --log-level DEBUG     --log-format json --input data\input\document.csv 


