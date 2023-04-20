from smart.auto import TreeMultiTask, AutoLoad

from huggingface_hub import snapshot_download

from llm_model import logger

auto_load = AutoLoad()

class HFModelTask(TreeMultiTask):
    Cache_Dir = '/data/share/model/huggingface'

    @auto_load.hook.before_task()
    def init_model(self, model_name=None, model_path=None, cache_dir=None):
        if not model_path:
            assert model_name, 'HFModelTask.init_model invalid args'
            cache_dir = cache_dir or self.Cache_Dir
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only = True
            )
            logger.debug("HFModelTask found model %s", model_path)
        
        self.context.state('llm_model_service').set(('hf_model', model_name), {
            'model_path': model_path
        })

        return {
            'model_name': model_name,
            'model_path': model_path
        }