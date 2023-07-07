from typing import List


def extract_parameters_from_comma_separated_string(comma_separated_parameters: str) -> List[str]:
    parameter_list = comma_separated_parameters.split(",")
    return parameter_list
