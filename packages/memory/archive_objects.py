"""Archive visible inputs and approved tool outputs, never evaluator objects."""


class ArchiveObjectReader:
    def __init__(self, visible, output):
        if visible.root == output.root:
            raise ValueError("memory_archive_object_separation")
        self.visible, self.output = visible, output

    def read(self, artifact):
        # Same collision rule as the original research Runtime's object reader.
        # Path validation is delegated to the stores, including URI/root checks.
        v = self.visible._path(artifact.uri).exists()
        o = self.output._path(artifact.uri).exists()
        if v and o:
            raise ValueError("memory_archive_object_collision")
        return (self.output if o else self.visible).read_bounded(artifact.uri, 4194304)
