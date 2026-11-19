"""Candidate-only public port; future generators must use a new advice version."""


class MemoryDistiller:
    def __init__(self, *, repository):
        self.repository = repository

    def prepare(self, *, request_key, episode_keys):
        return self.repository.prepare_distillation(request_key, episode_keys)

    def write(self, *, request_key, episode_keys):
        values = self.prepare(request_key=request_key, episode_keys=episode_keys)
        return self.repository.create_distillation(request_key, values)

    def read(self, key, *, historical=False):
        return self.repository.read(key, historical=historical)
