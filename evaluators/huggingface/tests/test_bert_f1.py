from langevals_huggingface.bert_f1 import (
    BertF1Evaluator,
    BertF1Entry,
    BertF1Settings,
)


def test_bert_f1_evaluator():
    entry = BertF1Entry(
        output="Mark Rutte is the president of the Netherlands",
        expected_output="Mark Rutte is the prime minister of the Neterlands.",
    )
    evaluator = BertF1Evaluator(settings=BertF1Settings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score > 0.8


def test_bert_f1_evaluator_lowers_f1_for_extra_text_in_output():
    entry = BertF1Entry(
        output="Mark Rutte is the president of the Netherlands",
        expected_output="Mark Rutte is the prime minister of the Neterlands, and I like bananas.",
    )
    evaluator = BertF1Evaluator(settings=BertF1Settings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score > 0.8

    entry = BertF1Entry(
        output="Mark Rutte is the president of the Netherlands, and I like bananas",
        expected_output="Mark Rutte is the prime minister of the Neterlands.",
    )
    evaluator = BertF1Evaluator(settings=BertF1Settings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score < 0.8
