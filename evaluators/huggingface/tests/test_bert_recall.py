from langevals_huggingface.bert_recall import (
    BertRecallEvaluator,
    BertRecallEntry,
    BertRecallSettings,
)


def test_bert_recall_evaluator():
    entry = BertRecallEntry(
        output="Mark Rutte is the president of the Netherlands",
        expected_output="Mark Rutte is the prime minister of the Neterlands.",
    )
    evaluator = BertRecallEvaluator(settings=BertRecallSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score > 0.8


def test_bert_recall_evaluator_lowers_recall_for_extra_text_in_output():
    entry = BertRecallEntry(
        output="Mark Rutte is the president of the Netherlands",
        expected_output="Mark Rutte is the prime minister of the Neterlands, and I like bananas.",
    )
    evaluator = BertRecallEvaluator(settings=BertRecallSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score < 0.78

    entry = BertRecallEntry(
        output="Mark Rutte is the president of the Netherlands, and I like bananas",
        expected_output="Mark Rutte is the prime minister of the Neterlands.",
    )
    evaluator = BertRecallEvaluator(settings=BertRecallSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score > 0.78
