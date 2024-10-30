import traceback

class Prompt:
    def __init__(
        self,
        question_prefix: str,
        answer_prefix: str,
        intra_example_sep: str,
        inter_example_sep: str,
        engine: str = None,
        temperature: float = None,
    ) -> None:
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.intra_example_sep = intra_example_sep
        self.inter_example_sep = inter_example_sep
        self.engine = engine
        self.temperature = temperature

    def make_query(self, prompt: str, question: str) -> str:
        return (
            f"{prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        )


def retry_parse_fail_prone_cmd(
    func,
    max_retries: int = 16,
    exceptions=(
        ValueError,
        KeyError,
        IndexError,
    ),
):
    def wrapper(*args, **kwargs):
        retries = max_retries
        while retries:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                stack_trace = traceback.format_exc()

                retries -= 1
                # print(f"An error occurred: {e}. {stack_trace}. Left retries: {retries}.")
                # print(f"... Left retries: {retries} ... {func.__name__} {args} {kwargs}")
        else:
            print(f"An error occurred: {e}. {stack_trace}. Running out of retries. Returning None as default value")
        return None

    return wrapper
