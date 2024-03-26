from langevals_huggingface.bert_precision import (
    BertPrecisionEvaluator,
    BertPrecisionEntry,
    BertPrecisionSettings,
)


def test_bert_precision_evaluator():
    entry = BertPrecisionEntry(
        output="Mark Rutte is the president of the Netherlands",
        expected_output="Mark Rutte is the prime minister of the Neterlands.",
    )
    evaluator = BertPrecisionEvaluator(settings=BertPrecisionSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score > 0.8


def test_bert_precision_evaluator_lowers_precision_for_extra_text_in_output():
    entry = BertPrecisionEntry(
        output="Mark Rutte is the president of the Netherlands",
        expected_output="Mark Rutte is the prime minister of the Neterlands, and I like bananas.",
    )
    evaluator = BertPrecisionEvaluator(settings=BertPrecisionSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score > 0.8

    entry = BertPrecisionEntry(
        output="Mark Rutte is the president of the Netherlands, and I like bananas",
        expected_output="Mark Rutte is the prime minister of the Neterlands.",
    )
    evaluator = BertPrecisionEvaluator(settings=BertPrecisionSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score < 0.8
