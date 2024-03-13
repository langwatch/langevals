from langevals_lingua.language_detection import (
    LinguaLanguageDetectionEvaluator,
    LinguaLanguageDetectionEntry,
    LinguaLanguageDetectionSettings,
)

# describe("LanguageCheck Integration", () => {
#   it("evaluates language check with a real request", async () => {
#     const sampleTrace: Trace = {
#       trace_id: "integration-test-language",
#       project_id: "integration-test",
#       metadata: {},
#       input: { value: "hello how is it going my friend? testing" },
#       output: { value: "ola como vai voce eu vou bem obrigado" },
#       metrics: {},
#       timestamps: { started_at: Date.now(), inserted_at: Date.now() },
#     };

#     const parameters: Checks["language_check"]["parameters"] = {
#       checkFor: "input_matches_output",
#       expectedLanguage: "any",
#     };

#     const result = await languageCheck(sampleTrace, [], parameters);

#     expect(result.status).toBe("failed");
#     expect(result.value).toBe(0);
#   });

#   it("evaluates language check with a specific expected language", async () => {
#     const sampleTrace: Trace = {
#       trace_id: "integration-test-language-specific",
#       project_id: "integration-test",
#       metadata: {},
#       input: { value: "hello how is it going my friend? testing" },
#       output: { value: "hello how is it going my friend? testing" },
#       metrics: {},
#       timestamps: { started_at: Date.now(), inserted_at: Date.now() },
#     };

#     const parameters: Checks["language_check"]["parameters"] = {
#       checkFor: "input_matches_output",
#       expectedLanguage: "EN",
#     };

#     const result = await languageCheck(sampleTrace, [], parameters);
#     expect(result.status).toBe("succeeded");
#     expect(result.value).toBe(1);
#   });

#   it("passes if it could not detect language", async () => {
#     const sampleTrace: Trace = {
#       trace_id: "integration-test-language-specific",
#       project_id: "integration-test",
#       metadata: {},
#       input: { value: "small text" },
#       output: { value: "small text" },
#       metrics: {},
#       timestamps: { started_at: Date.now(), inserted_at: Date.now() },
#     };

#     const parameters: Checks["language_check"]["parameters"] = {
#       checkFor: "input_matches_output",
#       expectedLanguage: "EN",
#     };

#     const result = await languageCheck(sampleTrace, [], parameters);
#     expect(result.status).toBe("succeeded");
#     expect(result.value).toBe(1);
#   });

#   it("should be okay if 'any' language is expected", async () => {
#     const sampleTrace: Trace = {
#       trace_id: "integration-test-language-specific",
#       project_id: "integration-test",
#       metadata: {},
#       input: { value: "small text" },
#       output: { value: "hello how is it going my friend? testing" },
#       metrics: {},
#       timestamps: { started_at: Date.now(), inserted_at: Date.now() },
#     };

#     const parameters: Checks["language_check"]["parameters"] = {
#       checkFor: "input_matches_output",
#       expectedLanguage: "any",
#     };

#     const result = await languageCheck(sampleTrace, [], parameters);
#     expect(result.status).toBe("succeeded");
#     expect(result.value).toBe(1);
#   });
# });


def test_language_detection_evaluator():
    entry = LinguaLanguageDetectionEntry(
        input="hello how is it going my friend? testing",
        output="ola como vai voce eu vou bem obrigado",
    )
    evaluator = LinguaLanguageDetectionEvaluator(
        settings=LinguaLanguageDetectionSettings(check_for="input_matches_output")
    )
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == False
    assert result.score == 2
    assert (
        result.details
        == "Input and output languages do not match. Input languages detected: EN. Output languages detected: PT"
    )


def test_language_detection_evaluator_specific_language():
    entry = LinguaLanguageDetectionEntry(
        input="hello how is it going my friend? testing",
        output="hello how is it going my friend? testing",
    )
    evaluator = LinguaLanguageDetectionEvaluator(
        settings=LinguaLanguageDetectionSettings(
            check_for="input_matches_output", expected_language="EN"
        )
    )
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == True
    assert result.score == 1
    assert (
        result.details == "Input languages detected: EN. Output languages detected: EN"
    )


def test_language_detection_evaluator_no_language_detected():
    entry = LinguaLanguageDetectionEntry(
        input="small text",
        output="small text",
    )
    evaluator = LinguaLanguageDetectionEvaluator(
        settings=LinguaLanguageDetectionSettings(
            check_for="input_matches_output", expected_language="EN"
        )
    )
    result = evaluator.evaluate(entry)

    assert result.status == "skipped"
    assert result.details == "Skipped because the input has less than 7 words"


def test_language_detection_evaluator_any_language():
    entry = LinguaLanguageDetectionEntry(
        input="small text",
        output="hello how is it going my friend? testing",
    )
    evaluator = LinguaLanguageDetectionEvaluator(
        settings=LinguaLanguageDetectionSettings(
            check_for="output_matches_language", expected_language=None
        )
    )
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == True
    assert result.score == 1
    assert result.details == "Languages detected: EN"
