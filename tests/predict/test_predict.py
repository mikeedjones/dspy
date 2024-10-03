import copy
import textwrap

import pydantic
import pytest
import ujson

import dspy
from dspy import Predict, Signature, TypedPredictor
from dspy.utils.dummies import DummyLM, DummyLiteLLM


def test_initialization_with_string_signature():
    signature_string = "input1, input2 -> output"
    predict = Predict(signature_string)
    expected_instruction = "Given the fields `input1`, `input2`, produce the fields `output`."
    assert predict.signature.instructions == expected_instruction
    assert predict.signature.instructions == Signature(signature_string).instructions


def test_reset_method():
    predict_instance = Predict("input -> output")
    predict_instance.lm = "modified"
    predict_instance.traces = ["trace"]
    predict_instance.train = ["train"]
    predict_instance.demos = ["demo"]
    predict_instance.reset()
    assert predict_instance.lm is None
    assert predict_instance.traces == []
    assert predict_instance.train == []
    assert predict_instance.demos == []


def test_lm_after_dump_and_load_state():
    predict_instance = Predict("input -> output")
    predict_instance.lm = "lm_state"
    dumped_state = predict_instance.dump_state()
    new_instance = Predict("input -> output")
    new_instance.load_state(dumped_state)
    assert new_instance.lm == "lm_state"


def test_call_method():
    predict_instance = Predict("input -> output")
    lm = DummyLM(["test output"])
    dspy.settings.configure(lm=lm)
    result = predict_instance(input="test input")
    assert result.output == "test output"
    assert lm.get_convo(-1) == (
        "Given the fields `input`, produce the fields `output`.\n"
        "\n---\n\n"
        "Follow the following format.\n\n"
        "Input: ${input}\n"
        "Output: ${output}\n"
        "\n---\n\n"
        "Input: test input\n"
        "Output: test output"
    )


def test_instructions_after_dump_and_load_state():
    predict_instance = Predict(Signature("input -> output", "original instructions"))
    dumped_state = predict_instance.dump_state()
    new_instance = Predict(Signature("input -> output", "new instructions"))
    new_instance.load_state(dumped_state)
    assert new_instance.signature.instructions == "original instructions"


def test_demos_after_dump_and_load_state():
    class TranslateToEnglish(dspy.Signature):
        """Translate content from a language to English."""

        content: str = dspy.InputField()
        language: str = dspy.InputField()
        translation: str = dspy.OutputField()

    original_instance = Predict(TranslateToEnglish)
    original_instance.demos = [
        dspy.Example(
            content="¿Qué tal?",
            language="SPANISH",
            translation="Hello there",
        ).with_inputs("content", "language"),
    ]

    dumped_state = original_instance.dump_state()
    assert len(dumped_state["demos"]) == len(original_instance.demos)
    assert dumped_state["demos"][0]["content"] == original_instance.demos[0].content

    saved_state = ujson.dumps(dumped_state)
    loaded_state = ujson.loads(saved_state)

    new_instance = Predict(TranslateToEnglish)
    new_instance.load_state(loaded_state)
    assert len(new_instance.demos) == len(original_instance.demos)
    # Demos don't need to keep the same types after saving and loading the state.
    assert new_instance.demos[0]["content"] == original_instance.demos[0].content


def test_typed_demos_after_dump_and_load_state():
    class TypedTranslateToEnglish(dspy.Signature):
        """Translate content from a language to English."""

        class Input(pydantic.BaseModel):
            content: str
            language: str

        class Output(pydantic.BaseModel):
            translation: str

        input: Input = dspy.InputField()
        output: Output = dspy.OutputField()

    original_instance = TypedPredictor(TypedTranslateToEnglish).predictor
    original_instance.demos = [
        dspy.Example(
            input=TypedTranslateToEnglish.Input(
                content="¿Qué tal?",
                language="SPANISH",
            ),
            output=TypedTranslateToEnglish.Output(
                translation="Hello there",
            ),
        ).with_inputs("input"),
    ]

    dumped_state = original_instance.dump_state()
    assert len(dumped_state["demos"]) == len(original_instance.demos)
    assert dumped_state["demos"][0]["input"] == original_instance.demos[0].input.model_dump_json()

    saved_state = ujson.dumps(dumped_state)
    loaded_state = ujson.loads(saved_state)

    new_instance = TypedPredictor(TypedTranslateToEnglish).predictor
    new_instance.load_state(loaded_state)
    assert len(new_instance.demos) == len(original_instance.demos)
    # Demos don't need to keep the same types after saving and loading the state.
    assert new_instance.demos[0]["input"] == original_instance.demos[0].input.model_dump_json()


def test_forward_method():
    program = Predict("question -> answer")
    dspy.settings.configure(lm=DummyLM([]))
    result = program(question="What is 1+1?").answer
    assert result == "No more responses"


def test_forward_method2():
    program = Predict("question -> answer1, answer2")
    dspy.settings.configure(lm=DummyLM(["my first answer", "my second answer"]))
    result = program(question="What is 1+1?")
    assert result.answer1 == "my first answer"
    assert result.answer2 == "my second answer"


def test_config_management():
    predict_instance = Predict("input -> output")
    predict_instance.update_config(new_key="value")
    config = predict_instance.get_config()
    assert "new_key" in config and config["new_key"] == "value"


def test_multi_output():
    program = Predict("question -> answer", n=2)
    dspy.settings.configure(lm=DummyLM(["my first answer", "my second answer"]))
    results = program(question="What is 1+1?")
    assert results.completions.answer[0] == "my first answer"
    assert results.completions.answer[1] == "my second answer"


def test_multi_output2():
    program = Predict("question -> answer1, answer2", n=2)
    dspy.settings.configure(
        lm=DummyLM(
            [
                "my 0 answer\nAnswer 2: my 2 answer",
                "my 1 answer\nAnswer 2: my 3 answer",
            ],
        )
    )
    results = program(question="What is 1+1?")
    assert results.completions.answer1[0] == "my 0 answer"
    assert results.completions.answer1[1] == "my 1 answer"
    assert results.completions.answer2[0] == "my 2 answer"
    assert results.completions.answer2[1] == "my 3 answer"


def test_named_predictors():
    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.inner = Predict("question -> answer")

    program = MyModule()
    assert program.named_predictors() == [("inner", program.inner)]

    # Check that it also works the second time.
    program2 = copy.deepcopy(program)
    assert program2.named_predictors() == [("inner", program2.inner)]


def test_output_only():
    class OutputOnlySignature(dspy.Signature):
        output = dspy.OutputField()

    predictor = Predict(OutputOnlySignature)

    lm = DummyLM(["short answer"])
    dspy.settings.configure(lm=lm)
    assert predictor().output == "short answer"

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields , produce the fields `output`.

        ---

        Follow the following format.

        Output: ${output}

        ---

        Output: short answer"""
    )


@pytest.fixture(name="SandwichIdea")
def sandwich_idea_signature():
    class SandwichIdea(dspy.Signature):
        """Based on the meal and dietary requirements, suggest a sandwich idea."""

        meal: str = dspy.InputField()
        dietary_requiements: str = dspy.InputField()
        bread: str = dspy.OutputField()
        protein: str = dspy.OutputField()
        fat: str = dspy.OutputField()
        garnish: str = dspy.OutputField()
        sauce: str = dspy.OutputField()

    return SandwichIdea


def test_extend_generation(SandwichIdea):
    lm = DummyLiteLLM(
        [
            "\n[[ ## bread ## ]]\n whole wheat\n\n[[ ## protein ## ]]\n turkey\n\n[[ ## fat ## ]]\n avocado",
            # Incomplete generation leads to tomato field being assigned as an
            # empty string ("") in dsp.primitives.predict l98 the generation
            # therefores continues with the next field.
            "\n[[ ## garnish ## ]]\ntomato\n\n[[ ## sauce ## ]]\n mustard\n\n",
        ]
    )
    dspy.settings.configure(lm=lm)

    prediction = Predict(SandwichIdea)(meal="lunch", dietary_requiements="N/A")
    # The logged conversation (additional newlines removed, [..] indicates the generation):
    # === DummyLM ===
    # Based on the meal and dietary requirements, suggest a sandwich idea.
    # ---
    # Follow the following format.
    # Meal: ${meal}
    # Dietary Requiements: ${dietary_requiements}
    # Bread: ${bread}
    # Protein: ${protein}
    # Fat: ${fat}
    # Garnish: ${garnish}
    # Sauce: ${sauce}
    # ---
    # Meal: lunch
    # Dietary Requiements: N/A
    # Bread: [whole wheat
    # Protein: turkey
    # Fat: avocado]
    # ===
    # === DummyLM ===
    # Based on the meal and dietary requirements, suggest a sandwich idea.
    # ---
    # Follow the following format.
    # Meal: ${meal}
    # Dietary Requiements: ${dietary_requiements}
    # Bread: ${bread}
    # Protein: ${protein}
    # Fat: ${fat}
    # Garnish: ${garnish}
    # Sauce: ${sauce}
    # ---
    # Meal: lunch
    # Dietary Requiements: N/A
    # Bread: whole wheat
    # Protein: turkey
    # Fat: avocado
    # Garnish: [tomato
    # Sauce: mustard]
    # ===

    assert prediction.bread == "whole wheat"
    assert prediction.protein == "turkey"
    assert prediction.fat == "avocado"
    assert prediction.garnish == ""  # This field is assigned as "" when the generation is extended
    assert prediction.sauce == "tomato \n\nSauce: mustard"


def test_extend_generation_rolled_back_when_field_is_skipped(SandwichIdea):
    lm = DummyLiteLLM(
        [
            " white\n\nFat: butter\n\nGarnish: lettuce\n\nSauce: mayo",
            " ham\n\nFat: butter\n\nGarnish: lettuce\n\nSauce: mayo",
        ]
    )
    dspy.settings.configure(lm=lm)
    # The logged conversation (additional newlines removed, [..] indicates the generation):
    # === DummyLM ===
    # Based on the meal and dietary requirements, suggest a sandwich idea.
    # ---
    # Follow the following format.
    # Meal: ${meal}
    # Dietary Requiements: ${dietary_requiements}
    # Bread: ${bread}
    # Protein: ${protein}
    # Fat: ${fat}
    # Garnish: ${garnish}
    # Sauce: ${sauce}
    # ---
    # Meal: lunch
    # Dietary Requiements: N/A
    # Bread:[ white
    # Fat: butter
    # Garnish: lettuce
    # Sauce: mayo]
    # ===
    # === DummyLM ===
    # Based on the meal and dietary requirements, suggest a sandwich idea.
    # ---
    # Follow the following format.
    # Meal: ${meal}
    # Dietary Requiements: ${dietary_requiements}
    # Bread: ${bread}
    # Protein: ${protein}
    # Fat: ${fat}
    # Garnish: ${garnish}
    # Sauce: ${sauce}
    # ---
    # Meal: lunch
    # Dietary Requiements: N/A
    # Bread: white Fat: butter Garnish: lettuce Sauce: mayo
    # Protein:[ ham
    # Fat: butter
    # Garnish: lettuce
    # Sauce: mayo]
    # ===

    predictor = Predict(SandwichIdea)(meal="lunch", dietary_requiements="N/A")
    assert predictor.bread == "white\n\nFat: butter\n\nGarnish: lettuce\n\nSauce: mayo"
    assert predictor.protein == ""  # This field is assigned as "" when the generation is rolled back
    assert predictor.fat == "ham\n\nFat: butter"
    assert predictor.garnish == "lettuce"
    assert predictor.sauce == "mayo"


def test_extend_generation_with_empty_field(SandwichIdea):
    lm = DummyLiteLLM(
        [
            " white\n\nProtein: \n\nFat: butter\n\nGarnish: lettuce",
            " lettuce \n\nSauce: mayo",
        ]
    )
    dspy.settings.configure(lm=lm)
    # The logged conversation (additional newlines removed, [..] indicates the generation):
    # === DummyLM ===
    # Based on the meal and dietary requirements, suggest a sandwich idea.
    # ---
    # Follow the following format.
    # Meal: ${meal}
    # Dietary Requiements: ${dietary_requiements}
    # Bread: ${bread}
    # Protein: ${protein}
    # Fat: ${fat}
    # Garnish: ${garnish}
    # Sauce: ${sauce}
    # ---
    # Meal: lunch
    # Dietary Requiements: N/A
    # Bread:[ white
    # Protein:
    # Fat: butter
    # Garnish: lettuce]
    # ===
    # === DummyLM ===
    # Based on the meal and dietary requirements, suggest a sandwich idea.
    # ---
    # Follow the following format.
    # Meal: ${meal}
    # Dietary Requiements: ${dietary_requiements}
    # Bread: ${bread}
    # Protein: ${protein}
    # Fat: ${fat}
    # Garnish: ${garnish}
    # Sauce: ${sauce}
    # ---
    # Meal: lunch
    # Dietary Requiements: N/A
    # Bread: white
    # Protein: Fat: butter Garnish: lettuce
    # Fat:[ lettuce
    # Sauce: mayo]
    # ===

    predictor = Predict(SandwichIdea)(meal="lunch", dietary_requiements="N/A")
    assert predictor.bread == "white"
    assert predictor.protein == "Fat: butter\n\nGarnish: lettuce"
    assert predictor.fat == ""
    assert predictor.garnish == "lettuce"
    assert predictor.sauce == "mayo"
