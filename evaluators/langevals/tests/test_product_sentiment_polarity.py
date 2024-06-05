import dotenv

dotenv.load_dotenv()

from langevals_langevals.product_sentiment_polarity import (
    ProductSentimentPolarityEvaluator,
    ProductSentimentPolarityEntry,
)


def test_product_sentiment_polarity_evaluator_pass():
    entry = ProductSentimentPolarityEntry(
        output="The iPhone 15 Pro stands out with its A17 Bionic chip, superior camera system with additional features, and ProMotion display offering a 120Hz refresh rate. It also boasts a more premium build with titanium edges. On the other hand, the iPhone 15, while impressive, does not match the Pro’s advanced hardware and features, making the Pro an unbeatable choice for tech enthusiasts."
    )
    evaluator = ProductSentimentPolarityEvaluator()
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 3
    assert result.passed == True
    assert result.details
    assert result.raw_response == "very_positive"


def test_product_sentiment_polarity_evaluator_fail():
    entry = ProductSentimentPolarityEntry(output="While the iPhone 15 Pro's titanium design is marketed as more durable, don't be fooled. It may resist minor scratches better than previous models, but it's still prone to dents and dings. Apple often exaggerates durability claims to justify high prices. Investing in a good case is still necessary; otherwise, you're just risking a fragile, overpriced gadget.")
    evaluator = ProductSentimentPolarityEvaluator()
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 0
    assert result.passed == False
    assert result.details
    assert result.raw_response == "very_negative"


def test_product_sentiment_polarity_evaluator_skipped_for_non_product_related_outputs():
    entry = ProductSentimentPolarityEntry(output="Is Man City better than Arsenal?")
    evaluator = ProductSentimentPolarityEvaluator()
    result = evaluator.evaluate(entry)

    assert result.status == "skipped"
