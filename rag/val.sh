# python -m rag.main --data-path ./dataset/ \
#     --eval medqa medmcqa pubmedqa mmlu_anatomy mmlu_clinical_knowledge mmlu_college_biology mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine \
#     --eval-dir ./evaluation/test_qa \
#     --results-dir ./rag_eval_results

python -m rag.main --data-path ./dataset/ \
    --eval medmcqa \
    --eval-dir ./evaluation/test_qa \
    --results-dir ./rag_eval_results