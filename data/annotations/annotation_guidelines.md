# Annotation Guidelines

This document outlines the guidelines for annotating data across four components: **Exploration**, **Reasoning**, **Hypothesizing** and **Verification**. Each section lists the relevant annotation fields, definitions, and usage instructions.

---

## 1. Exploration

Exploration refers to the process of collecting the datapoints. We distinguish between two types of exploration: *Strategic* and *Non-strategic*.

### Annotated Fields:
- **Strategic exploration**  
  *Datapoint differs from at least one of the previously collected datapoint in only one variable.*
  - (A=1, B=2, C=3) -> (A=1, B=2, C=4)

- **Non-strategic exploration**  
  *Datapoint differs from all previous datapoints in multiple variables* (**random exploration**) *or it is the same already collected datapoint* (**repeated exploration**).
  - (A=1, B=2, C=3) -> (A=1, B=2, C=3)
  - (A=1, B=2, C=3) -> (A=2, B=3, C=6)

---

## 2. Reasoning

Reasoning refers to identifying the cause effect of the change.

### Annotated Fields:
- **Verbalizing**  
  *Only describing the change in the observation.*
  - Variable D became 5.

- **Direction change**  
  *Identifying the effect in terms of direction.*
  - Variable D decreased.
  - Variable A was doubled.

- **Magnitude change**  
  *Identifying the effect in terms of magnitude.*
  - Variable D became 7, half of the value it had before.
  - Variable D went from 5 to 10.

We annotate the three types above, separately for the sets of input (A, B, C) and output (D) variables, and define following cause-effect categories:
### Metrics:


- **Qualitative effect**  
  *When either for the input or output variables, only the direction, and not the magnitude, have been identified.*
  - D increased when A was increased from to 6.
  - D doubled in size when A and B decreased.

- **Quantitative effect**  
  *When either for the input or output variables, both the direction and the magnitude have been identified.*
  - D increased from 1 to 2 when A was increased from 5 to 6.
  - D doubled to 10 when A and B decreased both from 8 to 6.
 
- **Verbalizing (Statement)**  
  *All other cases.*
  - Variable D was set to 5 when A is 6, B is 2, and C is 1.

---

## 3. Verification

Verification refers to ability to engage and/or verify hypotheses.

### Annotated Fields:
- **Verification thought**  
  *The reasoning trace contains verification step.*
  - The collected datapoint matches the hypothesis, so the hypothesis is accepted.

- **Computational verification**  
  *The verification is calculated numerically step by step.*
  - Let's test the hypothesis D = A + B + C by putting the values from the observation: 3 + 4 + 1 = 8, which is different from the observation of D = 6. Thus, the hypothesis is not correct.

- **Correct computation**  
  *The numerical computation is correct.*
  - **Example for correct:** Let's test the hypothesis D = A + B + C by putting the values from the observation: 3 + 4 + 1 = 8, which is different from the observation of D = 6. Thus, the hypothesis is not correct.
  - **Example for incorrect:** Let's test the hypothesis D = A + B + C by putting the values from the observation: 3 + 4 + 1 = 5, which is different from the observation of D = 6. Thus, the hypothesis is not correct.

- **Accepted hypohtesis**  
  *The agent states it accepts hypothesis or concludes the hypothesis is supported with the observation following the verification.*
  - Let's test the hypothesis D = A + B + C by putting the values from the observation: 3 + 4 + 1 = 8, which is the same as the observation of D = 8. Thus, the hypothesis is correct.

---

## 4. Hypothesizing

Hypothesizing refers to refining hypotheses on-the-fly.

### Annotated Fields:
- **Repeated hypothesis**  
  *Hypothesis already stated before and not refined.*

- **Numerical modification**  
  *Hypohtesis contains additional numerical term that adjusts previously proposed hypohtesis*
  - A + B -> (A + B) / 3

---

## Notes

- Exploration annottion has been done automatically, while the rest required manual annotation.

