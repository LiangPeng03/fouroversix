class _DummyContextManager:
    def __enter__(self): return self
    def __exit__(self, *args): return None

def parameter(*args, **kwargs):
    return lambda cls_or_func: cls_or_func

def method(*args, **kwargs):
    return lambda f: f

def enable_output(*args, **kwargs):
    return _DummyContextManager()

def run(*args, **kwargs):
    return _DummyContextManager()

class App:
    def __init__(self, *args, **kwargs): pass
    def cls(self, *args, **kwargs):
        return lambda cls: cls
    def method(self, *args, **kwargs):
        return lambda f: f
    def function(self, *args, **kwargs):
        return lambda f: f

class Image:
    @staticmethod
    def from_registry(*args, **kwargs):
        return Image()
    def entrypoint(self, *args, **kwargs):
        return self
    def apt_install(self, *args, **kwargs):
        return self
    def uv_pip_install(self, *args, **kwargs):
        return self
    def run_commands(self, *args, **kwargs):
        return self
    def add_local_dir(self, *args, **kwargs):
        return self
    def add_local_file(self, *args, **kwargs):
        return self
    def env(self, *args, **kwargs):
        return self
    def run_function(self, *args, **kwargs):
        return self
    def workdir(self, *args, **kwargs):
        return self
    def imports(self, *args, **kwargs):
        return _DummyContextManager()

class Volume:
    @staticmethod
    def from_name(*args, **kwargs):
        return None

class Secret:
    @staticmethod
    def from_name(*args, **kwargs):
        return None

def is_local() -> bool:
    return True
