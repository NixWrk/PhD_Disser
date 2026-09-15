from pathlib import Path
import json
root=Path('Colab Notebooks/cardiac_4d_segmentation').resolve();work=root/'results/old_verification_20260914'
cfg={'sequence_dir':str(work/'sequence_01'),'output_dir':str(work/'run_01'),'segmentation_python':str(root/'.venv-v2/Scripts/python.exe'),'analysis_python':r'C:\PC\Python\python.exe','expected_totalsegmentator_version':'2.18.0','gpu':'0','notebook':str(root/'21.08_Историческая_контрастная_КТ_объёмы_камер.ipynb')}
(work/'run_config.local.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
for name in ['README.md','21.00_Карта_4D_сердца_и_RR.md']:
 p=root/name;s=p.read_text(encoding='utf8')
 s+='\n\n## Исторические контрастные серии: подготовка проверки изменений объёма\n\nПо решению автора от 14.09.2026 исторические контрастные КТ обрабатываются\nотдельно от существующих результатов 21.03. Референсные маски предстоит\nсоздать; их отсутствие не блокирует автоматическую сегментацию и расчёт\nобъёмов, но пока исключает оценку точности. Первый прогон включает пилот\nна четырёх фазах и затем полную последовательность из 20 фаз.\nПостановка, этапы и ограничения зафиксированы в\n[протоколе 21.08](21.08_Историческая_контрастная_КТ_объёмы_камер.md).\n'
 p.write_text(s,encoding='utf8',newline='\n')
print('Local run config and series links prepared.')
