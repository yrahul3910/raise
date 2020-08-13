from interpret.sk import Rx


class ResultsInterpreter:
    """Interprets results saved by Experiment."""
    def __init__(self, files):
        """
        Initializes the interpreter.

        :param files: List of files
        """
        self.learners = []
        self.metrics = []
        self.result = {}
        self.files = files

    def _read_file(self, filename) -> dict:
        """
        Populates list of learners and metrics, returning the dict in the file read.

        :param filename: File to read
        :return: dict read
        """
        with open(filename, "r") as f:
            line = f.readline()
        line = eval(line)
        self.learners = list(line.keys())
        self.metrics = list(line[self.learners[0]].keys())
        return line

    def compare(self):
        """Compares results for each metric using a Scott-Knott test"""
        lines = {}

        # Combine results
        for file in self.files:
            line = self._read_file(file)
            for key in line.keys():
                lines[key + "-" + file.split("/")[-1]] = line[key].copy()

        result = {}

        for metric in self.metrics:
            print(metric)
            print("=" * len(metric))
            for learner in self.learners:
                for file in self.files:
                    # Add to result
                    key = learner + "-" + file.split("/")[-1]
                    result[key] = lines[key][metric]

            Rx.show(Rx.sk(Rx.data(**result)))
            print()
        self.result = result
