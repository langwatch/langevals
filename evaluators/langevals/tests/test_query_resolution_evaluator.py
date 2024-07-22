import dotenv

dotenv.load_dotenv()

from langevals_langevals.query_resolution_evaluator import (
    QueryResolutionConversationMessageEntry,
    QueryResolutionConversationEntry,
    QueryResolutionConversationSettings,
    QueryResolutionConversationResult,
    QueryResolutionConversationEvaluator
)


def test_query_resolution_conversation_evaluator_pass():
    response1 = QueryResolutionConversationMessageEntry(input="Hey, how are you?", output="Hello, I am an assistant and I don't have feelings")
    response2 = QueryResolutionConversationMessageEntry(input="Okay, is there a president in the Netherlands? Also, tell me what is the system of government in the Netherlands?", output="There is no president in the Netherlands. The system of government is constitutional monarchy.")
    conversation = QueryResolutionConversationEntry(conversation=[response1, response2])
    settings = QueryResolutionConversationSettings(model='openai/gpt-3.5-turbo-1106', max_tokens=10000)
    evaluator = QueryResolutionConversationEvaluator(settings=settings)
    result = evaluator.evaluate(conversation)

    assert result.status == "processed"
    assert result.score == 1.0
    assert result.passed == True
    assert result.details

def test_query_resolution_conversation_evaluator_fail():
    response1 = QueryResolutionConversationMessageEntry(input="Hey, how are you?", output="Hello, I am an assistant and I don't have feelings")
    response2 = QueryResolutionConversationMessageEntry(input="Okay, is there a president in the Netherlands? Also, what equals 2 + 2? How many paws does a standard dog have?", output="There is no president in the Netherlands.")
    conversation = QueryResolutionConversationEntry(conversation=[response1, response2])
    settings = QueryResolutionConversationSettings(model='openai/gpt-3.5-turbo-1106', max_tokens=10000)
    evaluator = QueryResolutionConversationEvaluator(settings=settings)
    result = evaluator.evaluate(conversation)

    assert result.status == "processed"
    assert result.score < 0.8
    assert result.passed == False
    assert result.details



def test_product_sentiment_polarity_evaluator_skipped_for_non_product_related_outputs():
    response1 = QueryResolutionConversationMessageEntry(input="", output="")
    response2 = QueryResolutionConversationMessageEntry(input="", output="")
    conversation = QueryResolutionConversationEntry(conversation=[response1, response2])
    settings = QueryResolutionConversationSettings(model='openai/gpt-3.5-turbo-1106', max_tokens=10000)
    evaluator = QueryResolutionConversationEvaluator(settings=settings)
    result = evaluator.evaluate(conversation)

    assert result.status == "skipped"
