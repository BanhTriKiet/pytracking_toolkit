import importlib
import os


class EnvSettings:
    def __init__(self):
        
        pytracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        self.results_path = '{}/tracking_results/'.format(pytracking_path)
        self.segmentation_path = '{}/segmentation_results/'.format(pytracking_path)
        self.network_path = '{}/networks/'.format(pytracking_path)
        self.result_plot_path = '{}/result_plots/'.format(pytracking_path)
        self.otb_path = ''
        self.nfs_path = ''
        self.uav_path = ''
        self.tpl_path = ''
        self.vot_path = 'D:/CacMonHoc/Nam4/KLTN/VOT/toolkit/vot_workspace/sequences'
        self.got10k_path = ''
        self.lasot_path = ''
        self.lasot_extension_subset_path = ''
        self.trackingnet_path = ''
        self.oxuva_path = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.votlt2019_path='D:/CacMonHoc/Nam4/KLTN/VOT/toolkit/vot_workspace/sequences'
        self.got_packed_results_path = ''
        self.got_reports_path = ''
        self.tn_packed_results_path = ''
        print("Loaded local.py - votlt2019_path:", self.votlt2019_path)

def create_default_local_file():
    comment = {'results_path': 'Where to store tracking results',
               'network_path': 'Where tracking networks are stored.'}

    path = os.path.join(os.path.dirname(__file__), 'local.py')
    with open(path, 'w') as f:
        settings = EnvSettings()

        f.write('from pytracking.evaluation.environment import EnvSettings\n\n')
        f.write('def local_env_settings():\n')
        f.write('    settings = EnvSettings()\n\n')
        f.write('    # Set your local paths here.\n\n')

        for attr in dir(settings):
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            attr_val = getattr(settings, attr)
            if not attr.startswith('__') and not callable(attr_val):
                if comment_str is None:
                    f.write('    settings.{} = \'{}\'\n'.format(attr, attr_val))
                else:
                    f.write('    settings.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))
        f.write('\n    return settings\n\n')


def env_settings():
    env_module_name = 'pytracking.evaluation.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.local_env_settings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        # Create a default file
        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. '
                           'Then try to run again.'.format(env_file))