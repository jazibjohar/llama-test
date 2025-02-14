# OpenAI Token Limits and Cost Analysis

## Model Token Limits (Input)

| Model                  | Input Token Limit | Output Token Limit | Total Token Limit |
|-----------------------|-------------------|-------------------|-------------------|
| GPT-4            | 128K             | 4K               | 128K             |

## Cost Analysis (per 1K tokens)

### GPT-4 Models
- GPT-4
  - Input: $0.01/1K tokens
  - Output: $0.03/1K tokens

## Cost Examples

### Example: Processing a 1000-word document
Approximate tokens: 1,500 tokens
- GPT-4 Turbo: $0.015 (input) + $0.045 (output) = $0.06
- GPT-3.5 Turbo: $0.0015 (input) + $0.003 (output) = $0.0045

## Token Usage Tips
1. 1 token ≈ 4 characters in English (e.g., "hello" = 1 token, "international" = 4 tokens)
2. 1 word ≈ 1.3 tokens (average)
3. 1 page (500 words) ≈ 750 tokens



## Curriculum Cost Analysis

### Total Curriculum (1,196,880 tokens) (1196K Tokens)

#### GPT-4 Cost Calculation
- Input Cost: $11.97 (1,196,880 tokens × $0.01/1K tokens)
- Output Cost (assuming full response): $35.91 (1,196,880 tokens × $0.03/1K tokens)
- Total Cost: $47.88