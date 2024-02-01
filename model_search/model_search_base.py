from pathlib import Path


class ModelSearchBase:
    """
    This class is used to search the log directories and version directories
    where 'tfevents' files are stored.
    """

    def __init__(self, start_path: Path):
        self.start_path = start_path
        self.log_dirs = self._get_log_dirs(start_path)

    def _get_log_dirs(self, start_path):
        """
        Recursively searches for 'tfevents' files in all subdirectories of
        start_path.  Adds the .parent.parent of each found 'tfevents' file to a
        list and returns it.

        :param start_path: The directory path to start the search from
        :return: A list of pathlib.Path objects pointing to the directories
            containing the 'tfevents' files
        """
        log_dirs = set()
        start_path = Path(start_path)

        # Walk through all subdirectories and files
        for path in start_path.rglob("*"):
            if path.is_file() and "tfevents" in path.name:
                log_dirs.add(path.parent.parent)

        log_dirs = list(log_dirs)
        log_dirs.sort()

        return log_dirs

    def _get_version_dirs(self, log_dir):
        """
        Returns a list of pathlib.Path objects pointing to the version
        directories in the given log directory. The 'best_model' directory is
        excluded.

        :param log_dir: The directory path to search for version directories
        :return: A list of pathlib.Path objects pointing to the version
            directories
        """
        version_dirs = []
        for version_dir in log_dir.iterdir():
            if version_dir.is_dir():
                version_dirs.append(version_dir)

        # if 'best_model' is already in the list, remove it
        version_dirs = [d for d in version_dirs if not d.name == "best_model"]

        return version_dirs
