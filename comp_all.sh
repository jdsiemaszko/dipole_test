python plotter.py config/config_remesh_AVRM.yaml 
python plotter.py config/config_merge_AVRM.yaml 
python plotter.py config/config_AVRM.yaml 
python plotter.py config/config_VRM.yaml 
python comp.py config/config_remesh_AVRM.yaml config/config_AVRM.yaml
python comp.py config/config_merge_AVRM.yaml config/config_AVRM.yaml
python comp.py config/config_remesh_AVRM.yaml config/config_VRM.yaml
python comp.py config/config_merge_AVRM.yaml config/config_VRM.yaml
python comp.py config/config_AVRM.yaml config/config_VRM.yaml