export type EvaluatorDefinition<T extends EvaluatorTypes> = {
  name: string;
  description: string;
  category: "quality" | "rag" | "safety" | "policy" | "other" | "custom";
  docsUrl?: string;
  isGuardrail: boolean;
  settings: {
    [K in keyof Evaluators[T]["settings"]]: {
      description?: string;
      default: Evaluators[T]["settings"][K];
    };
  };
  result: {
    score?: {
      description: string;
    };
    passed?: {
      description: string;
    };
  };
};

export type EvaluatorTypes = keyof Evaluators;

export type Evaluators = {
  "google_cloud/dlp_pii_detection": {
    settings: {
      info_types: {
        phone_number: boolean;
        email_address: boolean;
        credit_card_number: boolean;
        iban_code: boolean;
        ip_address: boolean;
        passport: boolean;
        vat_number: boolean;
        medical_record_number: boolean;
      };
      min_likelihood:
        | "VERY_UNLIKELY"
        | "UNLIKELY"
        | "POSSIBLE"
        | "LIKELY"
        | "VERY_LIKELY";
    };
    result: {
      score: {
        description: "Amount of PII detected, 0 means no PII detected";
      };
      passed: {
        description: "If true then no PII was detected, if false then at lease one PII was detected";
      };
    };
  };
  "custom/basic": {
    settings: {
      rules: {
        field: "input" | "output";
        rule:
          | "contains"
          | "not_contains"
          | "matches_regex"
          | "not_matches_regex";
        value: string;
      }[];
    };
    result: {
      score: {
        description: "Returns 1 if all rules pass, 0 if any rule fails";
      };
    };
  };
  "example/word_count": {
    settings: Record<string, never>;
    result: {
      score: {
        description: "How many words are there in the output, split by space";
      };
    };
  };
  "openai/moderation": {
    settings: {
      model: "text-moderation-stable" | "text-moderation-latest";
      categories: {
        harassment: boolean;
        harassment_threatening: boolean;
        hate: boolean;
        hate_threatening: boolean;
        self_harm: boolean;
        self_harm_instructions: boolean;
        self_harm_intent: boolean;
        sexual: boolean;
        sexual_minors: boolean;
        violence: boolean;
        violence_graphic: boolean;
      };
    };
    result: {
      score: {
        description: "The model's confidence on primary category where the input violates the OpenAI's policy. The value is between 0 and 1, where higher values denote higher confidence.";
      };
    };
  };
  "ragas/answer_relevancy": {
    settings: {
      model: "gpt-3.5-turbo-1106" | "gpt-4-1106-preview";
    };
    result: {};
  };
  "ragas/context_precision": {
    settings: {
      model: "gpt-3.5-turbo-1106" | "gpt-4-1106-preview";
    };
    result: {};
  };
  "ragas/context_recall": {
    settings: {
      model: "gpt-3.5-turbo-1106" | "gpt-4-1106-preview";
    };
    result: {};
  };
  "ragas/context_relevancy": {
    settings: {
      model: "gpt-3.5-turbo-1106" | "gpt-4-1106-preview";
    };
    result: {};
  };
  "ragas/faithfulness": {
    settings: {
      model: "gpt-3.5-turbo-1106" | "gpt-4-1106-preview";
    };
    result: {};
  };
  "azure/content_safety": {
    settings: {
      severity_threshold: number;
      categories: {
        Hate: boolean;
        SelfHarm: boolean;
        Sexual: boolean;
        Violence: boolean;
      };
      output_type: "FourSeverityLevels" | "EightSeverityLevels";
    };
    result: {
      score: {
        description: "The severity level of the detected content from 0 to 7. A higher score indicates higher severity.";
      };
    };
  };
  "azure/jailbreak": {
    settings: Record<string, never>;
    result: {
      passed: {
        description: "If true then no jailbreak was detected, if false then a jailbreak was detected";
      };
    };
  };
};

export const AVAILABLE_EVALUATORS: {
  [K in EvaluatorTypes]: EvaluatorDefinition<K>;
} = {
  "google_cloud/dlp_pii_detection": {
    name: `Google Cloud DLP PII Detection`,
    description: `
Google DLP PII detects personally identifiable information in text, including phone numbers, email addresses, and
social security numbers. It allows customization of the detection threshold and the specific types of PII to check.
`,
    category: "safety",
    docsUrl: "https://cloud.google.com/sensitive-data-protection/docs/apis",
    isGuardrail: true,
    settings: {
      info_types: {
        description: "The types of PII to check for in the input.",
        default: {
          phone_number: true,
          email_address: true,
          credit_card_number: true,
          iban_code: true,
          ip_address: true,
          passport: true,
          vat_number: true,
          medical_record_number: true,
        },
      },
      min_likelihood: {
        description:
          "The minimum confidence required for failing the evaluation on a PII match.",
        default: "POSSIBLE",
      },
    },
    result: {
      score: {
        description: "Amount of PII detected, 0 means no PII detected",
      },
      passed: {
        description:
          "If true then no PII was detected, if false then at lease one PII was detected",
      },
    },
  },
  "custom/basic": {
    name: `Custom Basic Evaluator`,
    description: `
Allows you to check for simple text matches or regex evaluation
`,
    category: "custom",
    docsUrl: "",
    isGuardrail: true,
    settings: {
      rules: {
        description: null,
        default: [],
      },
    },
    result: {
      score: {
        description: "Returns 1 if all rules pass, 0 if any rule fails",
      },
    },
  },
  "example/word_count": {
    name: `Example Evaluator`,
    description: `
This evaluator serves as a boilerplate for creating new evaluators.
`,
    category: "other",
    docsUrl: "https://path/to/official/docs",
    isGuardrail: false,
    settings: {},
    result: {
      score: {
        description: "How many words are there in the output, split by space",
      },
    },
  },
  "openai/moderation": {
    name: `OpenAI Moderation`,
    description: `
This evaluator uses OpenAI's moderation API to detect potentially harmful content in text,
including harassment, hate speech, self-harm, sexual content, and violence.
`,
    category: "policy",
    docsUrl: "https://platform.openai.com/docs/guides/moderation/overview",
    isGuardrail: true,
    settings: {
      model: {
        description:
          "The model version to use, `text-moderation-latest` will be automatically upgraded over time, while `text-moderation-stable` will only be updated with advanced notice by OpenAI.",
        default: "text-moderation-stable",
      },
      categories: {
        description: "The categories of content to check for moderation.",
        default: {
          harassment: true,
          harassment_threatening: true,
          hate: true,
          hate_threatening: true,
          self_harm: true,
          self_harm_instructions: true,
          self_harm_intent: true,
          sexual: true,
          sexual_minors: true,
          violence: true,
          violence_graphic: true,
        },
      },
    },
    result: {
      score: {
        description:
          "The model's confidence on primary category where the input violates the OpenAI's policy. The value is between 0 and 1, where higher values denote higher confidence.",
      },
    },
  },
  "ragas/answer_relevancy": {
    name: `Ragas Answer Relevancy`,
    description: `
This evaluator focuses on assessing how pertinent the generated answer is to the given prompt. Higher scores indicate better relevancy.
`,
    category: "rag",
    docsUrl:
      "https://docs.ragas.io/en/latest/concepts/metrics/answer_relevance.html",
    isGuardrail: false,
    settings: {
      model: {
        description: "The model to use for evaluation.",
        default: "gpt-3.5-turbo-1106",
      },
    },
    result: {},
  },
  "ragas/context_precision": {
    name: `Ragas Context Precision`,
    description: `
This metric evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Higher scores indicate better precision.
`,
    category: "rag",
    docsUrl:
      "https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html",
    isGuardrail: false,
    settings: {
      model: {
        description: "The model to use for evaluation.",
        default: "gpt-3.5-turbo-1106",
      },
    },
    result: {},
  },
  "ragas/context_recall": {
    name: `Ragas Context Recall`,
    description: `
This evaluator measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. Higher values indicate better performance.
`,
    category: "rag",
    docsUrl:
      "https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html",
    isGuardrail: false,
    settings: {
      model: {
        description: "The model to use for evaluation.",
        default: "gpt-3.5-turbo-1106",
      },
    },
    result: {},
  },
  "ragas/context_relevancy": {
    name: `Ragas Context Relevancy`,
    description: `
This metric gauges the relevancy of the retrieved context, calculated based on both the question and contexts. The values fall within the range of (0, 1), with higher values indicating better relevancy.
`,
    category: "rag",
    docsUrl:
      "https://docs.ragas.io/en/latest/concepts/metrics/context_relevancy.html",
    isGuardrail: false,
    settings: {
      model: {
        description: "The model to use for evaluation.",
        default: "gpt-3.5-turbo-1106",
      },
    },
    result: {},
  },
  "ragas/faithfulness": {
    name: `Ragas Faithfulness`,
    description: `
This evaluator assesses the extent to which the generated answer is consistent with the provided context. Higher scores indicate better faithfulness to the context.
`,
    category: "rag",
    docsUrl:
      "https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html",
    isGuardrail: false,
    settings: {
      model: {
        description: "The model to use for evaluation.",
        default: "gpt-3.5-turbo-1106",
      },
    },
    result: {},
  },
  "azure/content_safety": {
    name: `Azure Content Safety`,
    description: `
This evaluator detects potentially unsafe content in text, including hate speech,
self-harm, sexual content, and violence. It allows customization of the severity
threshold and the specific categories to check.
`,
    category: "safety",
    docsUrl:
      "https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-text",
    isGuardrail: true,
    settings: {
      severity_threshold: {
        description:
          "The minimum severity level to consider content as unsafe.",
        default: 1,
      },
      categories: {
        description: "The categories of moderation to check for.",
        default: {
          Hate: true,
          SelfHarm: true,
          Sexual: true,
          Violence: true,
        },
      },
      output_type: {
        description:
          "The type of severity levels to return on the full 0-7 severity scale, it can be either the trimmed version with four values (0, 2, 4, 6 scores) or the whole range.",
        default: "FourSeverityLevels",
      },
    },
    result: {
      score: {
        description:
          "The severity level of the detected content from 0 to 7. A higher score indicates higher severity.",
      },
    },
  },
  "azure/jailbreak": {
    name: `Azure Jailbreak Detection`,
    description: `
This evaluator checks for jailbreak-attempt in the input using Azure's Content Safety API.
`,
    category: "safety",
    docsUrl: "",
    isGuardrail: true,
    settings: {},
    result: {
      passed: {
        description:
          "If true then no jailbreak was detected, if false then a jailbreak was detected",
      },
    },
  },
};
