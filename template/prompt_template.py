# template/prompt_template.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 纯 Prompt 模板文件 — 只包含字符串常量，不包含任何业务逻辑/配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ==================== 基础推理 Prompts ====================

WEAK_BASELINE_TEST_PROMPT = '''
You are a mathematical problem solver.

Solve the question independently.

Rules:
1. Write the response in English only.
2. The first line must be exactly: Final Answer: \boxed{{actual_final_answer}}
3. You may give only 1-2 short lines of concise justification.
4. Keep the answer brief and direct.
5. The boxed content must contain exactly one final answer.

Question:
{question}

Your Solution:
'''


STRONG_GUIDED_TEST_PROMPT = '''
You are an expert mathematical problem solver.

Current problem surface semantics:
- core_skill: {core_skill}
- context_constraints: {context_constraints}
- task_objective: {task_objective}

[PRINCIPLE PACKAGE]:
{principle_block}


[Question]:
{question}


====================
INSTRUCTIONS & OUTPUT FORMAT:
Solve the problem concisely.

If a real retrieved principle package is provided, use it only as a verification aid or a corrective constraint during solving.
If no reliable principle package is provided, solve from first principles and do not claim to use a principle.

Write the response in this order:
1. A brief step-by-step solution.
2. At most one short sentence explaining whether the retrieved principle helped verify or correct the reasoning. If no reliable principle was retrieved, omit this sentence.
3. The final answer on the last line.


CRITICAL RULES:
1. Write the response in English only.
2. Do NOT guess the answer upfront. Think step-by-step first.
3. The very last line of your response MUST be exactly: Final Answer: \boxed{{actual_final_answer}}
4. The boxed content must contain exactly one final answer (e.g., a number, mathematical expression, or coordinate).
5. Prefer an exact final answer over a decimal approximation unless the question explicitly asks for a decimal approximation.
6. If the problem is multiple-choice, give the exact matching expression or value in the box, not an approximate decimal unless the choices are decimal.
7. Do NOT claim to execute or use a principle if the package is empty, missing, or says "No retrieved principle."
'''


# 直接回答问题的 Prompt，要求模型必须进行链式思考并得出最终数值结果，且输出格式严格规范，方便后续检索和分析。
DIRECT_ANSWER_PROMPT = '''
You are an expert mathematical problem solver.

Follow these rules strictly:
1. Write the entire response in English only.
2. Do not output Chinese, mixed-language text, corrupted text, random headers, or roleplay content.
3. Solve the problem directly from first principles. Do not mention retrieval, principles, or external guidance.
4. If the problem is underspecified, place the mathematically correct condition or range in the final box.
5. Derive the answer before stating it. Write one compact `Reasoning:` paragraph containing only the necessary derivation.
6. Do not state, preview, guess, or box a final answer before the derivation is complete.
7. After the derivation, write one short `Check:` sentence using at most one direct substitution or sanity check.
8. Use \\boxed{{}} exactly once in the entire response. It must contain one final answer only.
9. The very last non-empty line must be exactly: Final Answer: \\boxed{{your_answer}}
10. Do not write anything after the final-answer line.
11. The response must contain exactly these three sections in order: `Reasoning:`, `Check:`, and `Final Answer:`; keep the entire response under 250 words.
12. Do not restart, repeat, explore alternate full solutions, reinterpret the question, or invent additional constraints.
13. Prefer an exact final answer over a decimal approximation unless the question explicitly asks for a decimal approximation.

Question:
{question}
'''


GUIDED_ANSWER_PROMPT = '''
You are an expert mathematical problem solver.

[Question]
{question}

[Optional Past Experience]
{principle_text}

Rules:
1. Write the response in English only.
2. Solve the complete problem from first principles.
3. The past experience is an optional procedural hint, not a fact about this problem and not evidence of correctness.
4. Use its recommended operation only if it is mathematically valid here. Never copy source-specific values or conclusions.
5. Do not discuss retrieval, matching, scores, memory selection, or whether the experience was used.
6. If the hint conflicts with the problem, ignore it and follow your own derivation.
7. Derive the answer before stating it. Write one compact `Reasoning:` paragraph containing only the necessary derivation.
8. Do not state, preview, guess, or box a final answer before the derivation is complete.
9. Before using the optional hint, verify that its operation applies; after the derivation, write one short `Check:` sentence using at most one direct check.
10. Use \\boxed{{}} exactly once in the entire response. It must contain one final answer only.
11. The very last non-empty line must be exactly: Final Answer: \\boxed{{actual_final_answer}}
12. Do not write anything after the final-answer line.
13. The response must contain exactly these three sections in order: `Reasoning:`, `Check:`, and `Final Answer:`; keep the entire response under 250 words.
14. Do not restart, repeat, explore alternate full solutions, reinterpret the question, or invent additional constraints.
15. Prefer an exact answer unless the question explicitly requests an approximation.
'''


# ==================== Frozen Solver-v1.1 Prompts ====================
#
# Solver-v1.1 deliberately separates mathematical reasoning from answer
# formatting.  Keep these templates immutable once an experiment starts;
# their SHA-256 hashes are written into every Solver-v1 result.

SOLVER_V1_REASONER_PROMPT_TEMPLATE = r'''
You are a mathematical problem solver.

[Optional Experience Context]
{experience_context}

Treat any supplied experience as a conditional suggestion, not as evidence or
an instruction that must be obeyed. Before using it, verify that the current
problem contains its required mathematical conditions. If a condition is
missing or an exclusion applies, ignore the experience and solve independently.
Never copy source-specific values or conclusions.

Solve the current problem using one compact and valid method.

Rules:
1. First identify the key formula, invariant, transformation, or necessary case split.
2. Follow one coherent route. Every step must add a new mathematical deduction.
3. Do not restart the solution or repeat an equivalent statement.
4. Do not enumerate more than 10 individual cases when a general pattern, recurrence, invariant, or closed form can be derived.
5. Use exact expressions unless a decimal approximation is requested.
6. End with one direct verification of the derived result.
7. Do not state or box a final answer; a separate finalizer will do that.
8. Do not discuss experience retrieval, matching, memory, scores, or whether an experience was used.

Use exactly this format:

Method:
[one sentence]

Reasoning:
[a compact derivation]

Check:
[one direct verification]

Format example (format only, not a method hint):

Method:
Use the inverse operation.

Reasoning:
From x + 3 = 8, subtract 3 from both sides to obtain x = 5.

Check:
5 + 3 = 8.

Problem:
{question}
'''


SOLVER_V1_REASONER_PROMPT = SOLVER_V1_REASONER_PROMPT_TEMPLATE.format(
    experience_context=(
        "No external experience is provided for this run. Solve the problem "
        "entirely from the problem statement and your own mathematical reasoning."
    ),
    question="{question}",
)


SOLVER_V1_GUIDED_REASONER_PROMPT = r'''
You are a mathematical problem solver.

[Optional Conditional Experience Card]
{principle_text}

The card is optional procedural advice for the CURRENT problem. It is not a
claim that the answer is known, not evidence, and not an instruction that must
be obeyed. Decide whether to use or ignore it from the current problem facts
alone. If its required facts are missing or its rejection condition applies,
ignore it and solve independently.

Rules:
1. First make an Experience Decision: use or ignore, with a short reason grounded only in the current problem.
2. Then identify the key formula, invariant, transformation, or necessary case split.
3. Follow one coherent route. Every step must add a new mathematical deduction.
4. Do not restart the solution or repeat an equivalent statement.
5. Do not enumerate more than 10 individual cases when a general pattern, recurrence, invariant, or closed form can be derived.
6. Use exact expressions unless a decimal approximation is requested.
7. End with one direct verification of the derived result.
8. Do not state or box a final answer; a separate finalizer will do that.
9. Do not discuss retrieval, matching, memory IDs, scores, confidence, source problems, or whether an external system marked the card applicable.

Use exactly this format:

Experience Decision:
use | ignore — [brief reason based on current-problem facts]

Method:
[one sentence]

Reasoning:
[a compact derivation]

Check:
[one direct verification]

Problem:
{question}
'''


SOLVER_V1_FINALIZER_PROMPT = r'''
Extract the final mathematical answer from the proposed reasoning.

Rules:
1. Check that the answer addresses the original problem.
2. Do not continue or rewrite the derivation.
3. Do not introduce a new solution method.
4. Use exactly one \boxed{{}}.
5. Output exactly one line and nothing else.

Required format:
Final Answer: \boxed{{actual_answer}}

Problem:
{question}

Proposed reasoning:
{reasoner_output}
'''


BASELINE_TEST_PROMPT = '''
You are an expert mathematical problem solver.

Solve the question independently.

Rules:
1. Write the response in English only.
2. The first line must be exactly: Final Answer: \boxed{{actual_final_answer}}
3. After the first line, provide at most 3 short lines of concise justification.
4. Do not output hidden reasoning or placeholders.
5. The boxed content must contain exactly one final answer.

Question:
{question}

Your Solution:
'''

# ==================== 带 Ground Rules 的推理 Prompt ====================

INFERENCE_WITH_GROUND_RULES_PROMPT = '''
You are an expert problem solver. Use the guidance below as soft recommendations during your reasoning process.

{ground_rules}

Now solve the following problem while referring to the guidance when it is relevant.
IMPORTANT OUTPUT RULES:
1) You MUST output the final answer on the FIRST line, exactly in this format:
Final Answer: \boxed{{actual_final_answer}}
2) Do NOT output placeholders such as "your_final_answer".
3) Do NOT output <think>, chain-of-thought, or hidden reasoning.
4) After the first line, provide at most 3 short lines of concise justification.
5) Keep the whole response under 120 words.
6) If uncertain, still provide your best final answer in the required boxed format.
7) The boxed content must contain exactly ONE final answer only (one number or one simplest symbolic expression). Do NOT put multiple equations/items in the box.

**Question:**
{question}

**Your Solution:**
'''


PRINCIPLE_TO_RULES_PROMPT = '''
You are a mathematical reasoning protocol compiler.

Convert the retrieved principle package into a short list of optional solving suggestions that can help with the current question.

Requirements:
1. Output only JSON.
2. The `ground_rules` field must be a compact block of suggestive advice, using soft wording like "consider", "try", "it may help", "you can".
3. Use the retrieved principle only if it matches the current question's structure.
4. Do not restate the full question.

Question:
{question}

Retrieved principle package:
{principle_block}

Respond ONLY with JSON:
{{"ground_rules": "Consider ...\nIt may help to ...", "key_constraints": ["constraint1", "constraint2"]}}
'''

PRINCIPLE_GUIDED_TEST_PROMPT = '''
You are a math reasoning agent. You MUST solve the following problem by strictly executing the mandatory principle package below.

Current problem surface semantics:
- core_skill: {core_skill}
- context_constraints: {context_constraints}
- task_objective: {task_objective}

[MANDATORY PRINCIPLE PACKAGE]:
{principle_block}

[INSTRUCTION]:
Before you begin your calculation, you MUST write ">>> EXECUTING PRINCIPLE: [State how you will apply it here] <<<".
Then, proceed to solve the problem step-by-step.

Output Rules:
1. The final answer must appear on the first line as: Final Answer: \boxed{{actual_final_answer}}
2. The boxed content must contain exactly one final answer.
3. Do not output placeholders or hidden reasoning tags.

Question:
{question}
'''


SEMANTIC_SURFACE_EXTRACTION_PROMPT = """You are extracting the 3-field surface semantics used by the Mem retrieval system.

Your task is to convert a math problem into a retrieval-oriented English semantic signature that is:
- concrete enough to separate nearby problem types,
- short enough to embed well,
- aligned with the style already stored in memory.

Return valid JSON with EXACTLY these 3 keys:
1. "core_skill"
2. "context_constraints"
3. "task_objective"

Field definitions:
- "core_skill":
  The main reusable math operation, theorem, technique, or problem type.
  Prefer specific labels such as "division with remainder", "right triangle trigonometry",
  "permutations with restrictions", "work-rate equation", "binomial expansion",
  "similar triangles", "probability calculation".
  Do NOT use vague labels like "structured reasoning", "problem solving", "algebra", or "calculation"
  unless the problem truly cannot be narrowed further.

- "context_constraints":
  The concrete structural setup that controls retrieval.
  Keep the mathematically relevant objects, relationships, and constraints.
  Good examples:
  - "total quantity distributed equally among a fixed number of recipients"
  - "right angle at X, tangent ratio given, hypotenuse length known"
  - "six items divided into two groups of three with two specific items in the same group"
  - "circle inside another circle with center distance and radius condition"
  Do NOT turn this into a full solution plan.
  Do NOT collapse it into a generic phrase like "some quantities are given".

- "task_objective":
  The exact quantity, object, or conclusion to determine.
  Keep it short and specific, such as:
  - "remaining lollipops kept by Marie"
  - "length of XY"
  - "probability value"
  - "total number of valid departure sequences"
  - "coefficient of x^5"

Style rules:
- Write ALL values in English only.
- Output raw JSON only. No markdown fences. No explanation.
- The 3 fields must describe different aspects of the same problem.
- Preserve mathematically important nouns when they help retrieval; do not over-strip the story.
- Generalize raw numbers when possible, but preserve the structure they create.
- Each value should be a short phrase, not a paragraph.
- Prefer the style already seen in memory entries:
  "core_skill": "division with remainder"
  "context_constraints": "total of several quantities distributed equally among a fixed number of friends"
  "task_objective": "remaining lollipops kept by Marie"

Few-shot style examples:

Example 1
Problem: Marie has 75 raspberry lollipops, 132 mint lollipops, 9 blueberry lollipops, and 315 coconut lollipops. She distributes as many as possible equally among 13 friends. How many lollipops does she keep?
Output:
{{
  "core_skill": "division with remainder",
  "context_constraints": "total of several lollipop counts distributed equally among 13 friends",
  "task_objective": "remaining lollipops kept by Marie"
}}

Example 2
Problem: In triangle XYZ, angle X is 90 degrees, tan Z = sqrt(3), and YZ = 150. Find XY.
Output:
{{
  "core_skill": "right triangle trigonometry",
  "context_constraints": "right angle at X, tan Z given, hypotenuse length known",
  "task_objective": "length of XY"
}}

Example 3
Problem: Six trains are organized into two groups of three, with A and B in the same group. Find the number of departure sequences.
Output:
{{
  "core_skill": "permutations with restrictions",
  "context_constraints": "six items divided into two groups of three with two specific items in the same group",
  "task_objective": "total number of different departure sequences"
}}

Input problem:
{question}

Return ONLY this JSON schema:
{{
  "core_skill": "specific reusable math skill",
  "context_constraints": "concrete structural setup and constraints",
  "task_objective": "exact quantity or object to determine"
}}"""

# ==================== 任务描述生成 Prompt ====================

TASK_DESC_PROMPT = '''
Extract the core task description from the following question. Focus on the main objective, remove specific numbers and context details.

Question: {question}

Respond ONLY with JSON in this exact format (no markdown, no extra text):
{{"taskDescription": {{"description": "your concise abstract task description here"}}}}
'''

# ==================== 整体任务描述 Prompt (单向量检索消融) ====================

HOLISTIC_DESCRIPTION_PROMPT = """You are writing a single holistic retrieval key for a math problem.

Your task: describe the problem in ONE concise English sentence that captures ALL three aspects:
- The core mathematical skill or technique required
- The key structural constraints and relationships
- The exact objective or quantity to find

The description must be:
- Concrete enough to distinguish this problem type from nearby types
- Short enough to embed well (ideally 15-40 words)
- Written in English only
- A single sentence, no bullet points

Good examples:
- "division with remainder to find how many items remain after equal distribution among a fixed number of recipients"
- "right triangle trigonometry using tangent ratio and known hypotenuse to find the length of a specific side"
- "permutations with grouping restrictions counting departure sequences of items divided into teams"
- "probability calculation of a specific sum when rolling multiple dice with given constraints"

Return valid JSON with exactly this schema:
{{
  "description": "one sentence holistic retrieval key"
}}

Problem:
{question}

Return ONLY raw JSON. No markdown fences. No explanation."""

# ==================== 结构化任务描述 Prompt (用于记忆匹配) ====================

STRUCTURED_TASK_DESC_PROMPT = '''
You are an expert at abstracting math problems into reusable structural descriptors.

Analyze the following math problem and extract THREE orthogonal tags:

1. **method**: The core mathematical method or theorem needed.
   Examples: "vietas_formulas", "modular_arithmetic", "triangle_inequality", "binomial_expansion", "lagrange_multipliers", "telescoping_product", "pigeonhole_principle", "completing_the_square"

2. **objective**: What the problem asks you to find or prove.
   Examples: "find_minimum", "find_count", "find_sum", "find_ratio", "determine_value", "prove_inequality", "find_remainder", "find_gcd"

3. **structure**: The mathematical object or structure involved.
   Examples: "arithmetic_sequence", "geometric_sequence", "polynomial", "quadratic_equation", "matrix", "triangle", "circle", "permutation", "recurrence_relation", "modular_system"

Rules:
- Use lowercase_snake_case for all tags.
- Each tag should be 1-3 words, highly reusable across similar problems.
- Be as SPECIFIC as the problem warrants, but prefer COMMON terms over rare ones.
- If unsure between two options, pick the more general one.

Problem:
{question}

Respond ONLY with JSON (no markdown):
{{"method": "...", "objective": "...", "structure": "..."}}
'''

# ==================== 第一层分类：问题 -> 六大类 (强模型领域分类器) ====================

QUESTION_CLASSIFICATION_PROMPT = '''
You are an expert mathematical domain classifier.
Classify the following math problem into exactly ONE of these six categories:

1. Numerical_Algebraic_Operations  — arithmetic, algebra, number theory, polynomials
2. Functions_Equations             — functions, equations, inequalities, limits, calculus
3. Trigonometry_Transformations    — trigonometric functions, identities, angle transformations
4. Geometry_Stereometry            — plane geometry, solid geometry, coordinate geometry
5. Linear_Algebra_Matrices         — vectors, matrices, determinants, linear systems
6. Combinatorics_Logic             — combinatorics, probability, logic, set theory

Problem:
{question}

Respond ONLY with JSON (no markdown):
{{"reasoning": "brief explanation of why this category", "category": "exact category name from the list above"}}
'''

# ==================== 第一层分类（备用）：任务描述 -> 六大类 ====================

CATEGORY_CLASSIFICATION_PROMPT = '''
You are an expert in mathematical problem classification. Your task is to classify a given task description into ONE of the following six categories:

Categories:
1. Numerical_Algebraic_Operations
2. Functions_Equations
3. Trigonometry_Transformations
4. Geometry_Stereometry
5. Linear_Algebra_Matrices
6. Combinatorics_Logic

Task: {task_description}

Respond ONLY with JSON (no markdown):
{{"reasoning": "brief explanation", "category": "exact category name"}}
'''

# ==================== 第二层匹配：类目内任务匹配 ====================

TASK_MATCHING_PROMPT = '''
Find the MOST semantically similar task from the existing list. You MUST choose one - find the closest match based on the core objective and problem type.

Existing tasks:
{existing_tasks}

New task: {new_task_desc}

Respond ONLY with JSON (no markdown):
{{"reasoning": "why this is the best match", "best_match": "copy exact task text from existing list"}}
'''

# ==================== 原则冲突检测与合并 ====================

PRINCIPLE_CONFLICT_PROMPT = '''
You are an expert in knowledge management. Compare new principles against existing principles and determine how to merge them.

Merge new principles with existing ones. Remove redundant, resolve conflicts, add novel.

Existing: {existing_principles}

New: {new_principles}

Respond ONLY with JSON (no markdown):
{{"merged_principles": ["principle1", "principle2"], "actions_taken": ["action1", "action2"]}}
'''

# ==================== Ground Rules 生成 ====================

GROUND_RULES_PROMPT = '''
Convert principles into mandatory ground rules for problem-solving.

Principles: {principles}

Context: {task_context}

Respond ONLY with JSON (no markdown):
{{"ground_rules": "imperative rules using MUST/SHALL/NEVER", "key_constraints": ["constraint1", "constraint2"]}}
'''

# ==================== 差异分析 Prompt ====================

# Stage 1 简化版: 直接对比两条轨迹,无需语义分解字段
STAGE1_DIFF_PROMPT = '''You are a strict reasoning analyst. Compare the correct trajectory and the wrong trajectory for the problem below. Identify the exact point where the wrong trajectory diverged and why it led to a wrong answer.

[Question]:
{question}

[Correct Trajectory]:
{chosen_path}

[Wrong Trajectory]:
{rejected_path}

{divergence_hint}

RULES:
1. Quote verbatim from the actual trajectories above. Do NOT fabricate or paraphrase.
2. Do NOT wrap the JSON in ``` fences. Do NOT add any text before or after the JSON.
3. Your entire response must start with {{ and end with }}.

CORRECT (do this):
{{"differences":[{{"Aspect":"Arithmetic Mistake","DivergenceStep":"x=(-3+5)/2=4","CorrectThinking":"x=(-3+5)/2=1","WrongThinking":"x=(-3+5)/2=4","CausalError":"5+(-3) was computed as 8 instead of 2","WhyItMatters":"Wrong x propagates through remaining steps"}}]}}

WRONG (do NOT do this):
```json
{{"differences":[...]}}
```
or: Here is the analysis: {{"differences":[...]}}

Now output:
{{
  "differences": [
    {{
      "Aspect": "e.g., Wrong Formula, Constraint Omission, Arithmetic Mistake, Logical Leap, Misinterpretation",
      "DivergenceStep": "Quote exact sentence from Wrong Trajectory where reasoning first went wrong",
      "CorrectThinking": "What the Correct Trajectory did at the same point",
      "WrongThinking": "What the Wrong Trajectory did instead",
      "CausalError": "Root cause connecting this mistake to the final wrong answer",
      "WhyItMatters": "Why this specific error, in this context, made the answer incorrect"
    }}
  ]
}}'''

# 完整版 (向后兼容,用于 Stage 2 principle extraction)
DIFF_PROMPT = '''You are a strict reasoning analysis bot. Compare the "Successful Path" and "Failed Path" for the problem below.

[Problem Essence]
- Core Skill: {core_skill}
- Context & Constraints: {context_constraints}
- Task Objective: {task_objective}
{divergence_hint}
[Question]:
{question}

[Successful Path]:
{chosen_path}

[Failed Path]:
{rejected_path}

====================
INSTRUCTIONS:
To analyze the error, you MUST structure your internal thinking based directly on the [Problem Essence] provided above. Follow these problem-specific steps:

**Step 1: Skill Alignment:** Analyze how the Failed Path handled the core skill of "[{core_skill}]". Did it apply the wrong method or reasoning logic related to this specific skill?
**Step 2: Constraint Validation:** Trace the constraints "[{context_constraints}]" in both paths. At what exact step did the Failed Path ignore, alter, or misapply these specific conditions?
**Step 3: Objective Deviation:** Evaluate if the Failed Path lost track of the ultimate goal: "[{task_objective}]". Did the errors in Step 1 or 2 cause it to solve for the wrong target?
**Step 4: Final Extraction:** Synthesize the exact point of collapse into the required JSON format.

CRITICAL RULES:
1. Do NOT output Step 1/2/3/4 as plain text. Use them ONLY as your internal reasoning process.
2. You MUST output ONLY valid JSON. No conversational text, no markdown code blocks around the JSON, just the raw JSON string starting with {{.
3. Use the exact keys shown in the schema below.

JSON SCHEMA:
{{
  "differences": [
    {{
      "Aspect": "Which essence was violated (e.g., Skill Misapplication, Constraint Omission, Objective Deviation)",
      "DivergenceStep": "Quote the EXACT sentence or line (verbatim) from the Failed Path where reasoning first went wrong",
      "CorrectThinking": "What the successful path did correctly at that divergence point, referencing the specific constraint or condition from the problem",
      "WrongThinking": "What the failed path did wrong at that divergence point, naming the specific object or concept involved",
      "CausalError": "The causal reasoning error that connects the wrong step to the final failure",
      "WhyItMatters": "Explain concretely—referencing the problem's specific conditions ({context_constraints})—why this mistake led to an incorrect result"
    }}
  ]
}}'''

# ==================== 原则生成 Prompt ====================

# 向后兼容的原则生成 Prompt（仅需 question + diff_list，用于 plan_module / engine 调用）
PRINCIPLE_PROMPT = '''You are a Formal Logic and Mathematical Reasoning Expert performing "Meta-Abstraction".

[Question]:
{question}

[Reasoning Collapse Point] (identified mistake):
{diff_list}

====================
Convert the specific error into ONE universal, actionable reasoning rule.

CRITICAL RULES:
1. NO GENERIC ADVICE (e.g., "be careful", "check your logic").
2. MUST BE ACTIONABLE: dictate a concrete mathematical action.
3. JSON OUTPUT ONLY, starting with {{.

JSON SCHEMA:
{{
  "principle": "A single imperative sentence with a concrete mathematical action.",
  "explanation": "How this action prevents the identified mistake."
}}'''

STRUCTURED_PRINCIPLE_PROMPT = '''You are a Formal Logic and Reasoning Expert. Your task is "Meta-Abstraction": converting specific error observations into highly condensed, reusable, and generalized reasoning principles.

[Problem Essence]
- Core Skill: {core_skill}
- Context Constraints: {context_constraints}
- Task Objective: {task_objective}

[Question]:
{question}

[Reasoning Collapse Point] (Difference List):
{diff_list}

[Validation Feedback]:
{feedback_block}

====================
INSTRUCTIONS:
To perform Meta-Abstraction, you MUST structure your internal thinking using the following steps based on the data above:

**Step 1: Root Cause Mapping:** Analyze the [Reasoning Collapse Point]. Which part of the [Problem Essence] was violated? 
**Step 2: De-contextualization:** Strip away the specific numbers, variables, or story elements from the [Question]. Elevate the specific error to a structural reasoning level.
**Step 3: Actionable Rule Generation:** Formulate EXACTLY 3 candidate principles. Each must be a strict, executable command that forces a problem solver to take a concrete action to avoid this trap in future problems.

CRITICAL RULES:
1. Do NOT output Step 1/2/3 as plain text. Use them ONLY as your internal reasoning process.
2. NO GENERIC ADVICE: Never use useless words like "be careful", "check boundary conditions", or "ensure logical flow".
3. ACTIONABLE OPERATOR: Provide a concrete, reproducible action (e.g., "Explicitly establish domain boundaries before squaring any equation").
4. ANTI-CHEAT (NO HALLUCINATION): DO NOT copy exact numbers or specific expressions from the prompt. Teach the METHOD.
5. REQUIRED EVIDENCE: State the observable structural facts that must all hold before the action is used.
6. EXCLUSION: State a near-neighbor mismatch that forces abstention.
7. PARAMETERIZATION: Never place a source-instance coefficient or formula in `action_guidance` unless it is a universal identity.
8. FAMILY-LEVEL TRIGGER: Describe the minimum mathematical structure needed by the action. Do not require the source problem's numbers, variable names, story, or exact requested quantity when the same operation remains executable.
9. SEPARATION OF DUTIES: Keep the trigger recall-oriented; place mandatory facts in `required_evidence` and decisive near-neighbor mismatches in `exclusion_condition`.
10. JSON OUTPUT ONLY: You MUST output ONLY valid JSON. No conversational text, no markdown blocks around the JSON, just the raw JSON string starting with {{.

JSON SCHEMA:
{{
  "principles": [
    {{
      "introspection": "[Why the model failed at the divergence point]",
      "trigger_condition": "[A concise condition that should activate the principle]",
      "required_evidence": ["[One observable mandatory fact]", "[Another observable mandatory fact]"],
      "exclusion_condition": "[Observable mismatch that forces rejection]",
      "action_guidance": "[A concrete, actionable reasoning rule in ONE imperative sentence explaining exactly what step to take]",
      "verification_step": "[A concrete check after applying the action]",
      "generalization_level": "instance | family | universal",
      "scope": "[The class of problems this principle applies to]",
      "failure_mode": "[The specific failure pattern this principle prevents]"
    }}
  ]
}}'''
# ==================== 诊断原则提取 Prompt ====================

DIAGNOSTIC_PRINCIPLE_EXTRACT_PROMPT = '''
You are a senior mathematical reasoning reviewer with an omniscient perspective.
Given the question, the three-dimensional feature signature, and both correct (chosen) and incorrect (rejected) answers,
generate a "diagnostic principle" - a specific pitfall-avoidance guide for this题型.

Input:
1. Question: {question}
2. Feature Signature:
   - core_skill: {core_skill}
   - context_constraints: {context_constraints}
   - task_objective: {task_objective}
3. Correct Answer (chosen): {chosen}
4. Incorrect Answer (rejected): {rejected}

Task:
Analyze why the correct answer succeeds and the incorrect answer fails.
Generate EXACTLY ONE diagnostic principle that captures the key pitfall to avoid.

Requirements:
1. NO GENERIC ADVICE: Avoid words like "careful", "check carefully", "be mindful"
2. MUST reference specific structural elements or constraints from the feature signature
3. Include the specific trap or error pattern that distinguishes chosen from rejected
4. Make it actionable - tell the solver EXACTLY what to verify or consider

Output JSON:
{{"principle": "...", "application": "..."}}
'''

# ==================== 记忆合并路由 Prompt (四态演进) ====================

MEMORY_MERGE_ROUTER_PROMPT = '''
You are a knowledge integration expert. Given existing principles and a new principle,
determine how to evolve the knowledge base.

Existing principles for this feature node:
{existing_principles}

New principle to merge:
{new_principle}

Analyze the relationship and output your decision:

Rules:
- APPEND: Add as a new, independent考点 that wasn't covered before
- UPDATE: Replace a similar but less accurate principle with the new one
- RESOLVE_CONFLICT: Rewrite contradictory principles as an If-Else conditional branch
- IGNORE: The new principle is redundant with existing ones

Respond ONLY with JSON:
{{
  "action": "APPEND | UPDATE | RESOLVE_CONFLICT | IGNORE",
  "reasoning": "brief explanation",
  "updated_principles": [...]  // Final list after action
}}
'''

# ==================== Stage 2 Self-CPE Prompt (Self-Contrastive Principle Extraction) ====================

# ==================== Stage 3 Memory Applicability Gate ====================

MEMORY_APPLICABILITY_PROMPT = '''You are a conservative structural routing judge for a math-reasoning memory system.
The retrieval stage produced candidate memories. Compare each memory's SOURCE problem structure against the CURRENT problem before the solver sees any guidance.

Rules:
1. Judge only applicability; do not solve the current problem and do not rewrite the action.
2. Topic or vocabulary similarity is insufficient. Compare mathematical objects, recurrences/invariants, constraints, objective, and the operation required by the action.
3. `trigger_match=true` only when the stated trigger is observable in the current problem.
4. `required_evidence_match=true` only when ALL required evidence is present.
5. `structural_match=true` when the mathematical operator or invariant needed by the action is preserved after rebinding roles. A fixed row width and a doubling row width are NOT a match.
6. `scope_match=true` when the current problem belongs to the same mathematical family. Different numbers, variable names, story wording, or a reversed unknown do not alone force rejection if the action is still executable.
7. `contraindicated=true` when an exclusion condition holds or a source-specific constant/formula would be transferred without being derivable from the current problem.
8. Reject uncertain candidates. Do not accept merely because both problems use the same named skill, but do not reject merely because source-instance constants differ.
9. Confidence is the probability that the memory is structurally applicable. Rejected candidates should normally be below 0.70.
10. Check elementary mathematical claims before using them as a rejection reason (for example, whether a power is a perfect square or cube).
11. Return one judgment for every candidate ID inside the documented top-level object and output raw JSON only.

[Current Problem]
{question}

[Candidate Routing Fields]
{candidates_json}

JSON SCHEMA:
{{
  "judgments": [
    {{
      "memory_id": "candidate id",
      "trigger_match": true,
      "required_evidence_match": true,
      "structural_match": true,
      "scope_match": true,
      "contraindicated": false,
      "violated_conditions": ["short observable mismatch"],
      "source_specific_elements": ["constant/formula tied to the source problem"],
      "confidence": 0.0,
      "reason": "one short routing reason"
    }}
  ]
}}'''


MEMORY_ACTION_BINDING_PROMPT = '''You are a constrained math-memory adapter.
A separate structural gate has already accepted this memory for the current problem. Rewrite the accepted memory into ONE six-field conditional card for the CURRENT problem without solving the problem.

Rules:
1. Preserve the mathematical intent of the original memory, but write only current-problem-facing guidance.
2. Remove source questions, source answers, memory IDs, retrieval scores, confidence, source-specific numbers, source variable names, source-only conclusions, and formulas unless the same value or formula is explicitly derivable from the current problem statement.
3. Bind each required evidence item to an observable current-problem fact. If a requirement cannot be matched to the current problem, return `status="reject"`.
4. Do not add a new theorem, fact, assumption, intermediate result, or answer.
5. Do not solve the problem and do not reveal any final answer.
6. `suggested_action` should be 3-6 concise advisory sentences. It may explain why the action helps, but it must not assert that the card is reliable or applicable.
7. `verification` should contain 2-4 concrete checks the solver can perform after using the action.
8. If safe adaptation is impossible, return `status="reject"` and leave all card fields empty except `reason`.
9. Output raw JSON only.

[Current Problem]
{question}

[Accepted Source Memory]
{candidate_json}

JSON SCHEMA:
{{
  "status": "use | reject",
  "when_to_consider": "current-problem condition under which the solver should consider the card",
  "evidence_found": [
    {{"requirement": "required evidence from the memory", "problem_fact": "matching fact observed in the current problem"}}
  ],
  "do_not_use_if": "current-problem condition that should make the solver ignore this card",
  "failure_to_avoid": "short current-problem-facing description of the mistake to avoid",
  "suggested_action": "3-6 advisory executable sentences, source-free and answer-free",
  "verification": "2-4 current-problem checks",
  "reason": "why the card can or cannot be safely rewritten"
}}
'''

# ==================== Stage 2 Repair Diagnosis Prompt ====================

REPAIR_DIAGNOSIS_PROMPT = '''You are a principle quality auditor. A generated reasoning principle has FAILED validation on neighbor tasks — it did not help the model solve problems with similar logical structure.

Your job: diagnose WHY the principle failed and classify the failure mode.

[The Principle Under Audit]:
- Introspection: {introspection}
- Trigger Condition: {trigger_condition}
- Required Evidence: {required_evidence}
- Exclusion Condition: {exclusion_condition}
- Action Guidance: {action_guidance}
- Verification Step: {verification_step}
- Generalization Level: {generalization_level}
- Scope: {scope}
- Failure Mode: {failure_mode}

[Failed Validation Cases]:
The principle was tested as guidance on the following neighbor problems, but the model still answered incorrectly:
{failed_cases}

[Source Problem Context]:
{source_context}

====================
DIAGNOSIS TYPES:

**"vague"**: The principle is too generic. The action guidance lacks specificity — the model cannot translate it into a concrete step. E.g., "verify constraints" instead of "substitute each variable into the original inequality to check sign consistency."

**"over_general"**: The trigger condition or scope is too broad. The principle fires on problems where it does not apply, causing the model to follow incorrect guidance. E.g., a principle about "always check extraneous solutions" triggering on linear equations.

**"misdirected"**: The principle addresses a symptom rather than the root cause. The action guidance tells the model to check something that would not have prevented the actual error.

**"other"**: The failure is not clearly classifiable. The principle may be fundamentally incorrect or the neighbor tasks may not be truly isomorphic.

====================
INSTRUCTIONS:
1. Read the principle and the failed cases carefully.
2. Identify the pattern: did the principle fail because it was too weak (vague), too broad (over_general), aimed at the wrong thing (misdirected), or something else?
3. Output your diagnosis as raw JSON only.

JSON SCHEMA:
{{
  "diagnosis": "vague | over_general | misdirected | other",
  "reason": "[2-3 sentences explaining the diagnosis, citing specific evidence from the failed cases]",
  "suggested_fix_direction": "[1-2 sentences describing what kind of fix would address this — e.g., add a concrete verification step with explicit variables, or narrow scope to polynomial equations only]"
}}'''

# ==================== Stage 2 Repair Patch Prompt ====================

REPAIR_PATCH_PROMPT = '''You are a principle repair specialist. A principle has been diagnosed as flawed, and you must rewrite it to fix the specific issue.

[Original Principle]:
- Introspection: {introspection}
- Trigger Condition: {trigger_condition}
- Required Evidence: {required_evidence}
- Exclusion Condition: {exclusion_condition}
- Action Guidance: {action_guidance}
- Verification Step: {verification_step}
- Generalization Level: {generalization_level}
- Scope: {scope}
- Failure Mode: {failure_mode}

[Diagnosis]:
- Failure Type: {diagnosis}
- Reason: {reason}
- Suggested Fix: {suggested_fix_direction}

[Failed Example]:
{failed_example}

[Source Problem Essence]:
- Core Skill: {core_skill}
- Context Constraints: {context_constraints}
- Task Objective: {task_objective}

====================
REPAIR INSTRUCTIONS:

If diagnosis is **"vague"**: Make the action guidance CONCRETE. Replace general verbs with specific operations. Add a concrete verification step with explicit reference to the variables or structures involved. The model should be able to execute the guidance without interpretation.

If diagnosis is **"over_general"**: NARROW the trigger condition and scope. Add specific qualifiers that restrict when this principle should activate. Think: what must be TRUE about the problem for this principle to be useful?

If diagnosis is **"misdirected"**: REWRITE the introspection and action guidance to target the ROOT CAUSE, not the symptom. What should the model have done DIFFERENTLY at the divergence point?

CRITICAL RULES:
1. Output the REPAIRED principle with the SAME JSON schema.
2. The repaired principle MUST differ meaningfully from the original. If you cannot fix it, output the original unchanged but with status "unfixable".
3. JSON OUTPUT ONLY.

JSON SCHEMA:
{{
  "principle": {{
    "introspection": "[Root cause — why the model fails]",
    "trigger_condition": "[When to activate]",
    "required_evidence": ["[One fact that must be observed before activation]"],
    "exclusion_condition": "[Mismatch that forces rejection]",
    "action_guidance": "[Concrete step to take]",
    "verification_step": "[Concrete validity check after applying the action]",
    "generalization_level": "instance | family | universal",
    "scope": "[Applicable problem class]",
    "failure_mode": "[What this prevents]",
    "repair_status": "repaired | unfixable"
  }}
}}'''

