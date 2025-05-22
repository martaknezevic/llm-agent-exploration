# Annotation Guidelines

This document outlines the guidelines for annotating data across four components: **Exploration**, **Reasoning**, **Hypothesizing** and **Verification**. Each section lists the relevant annotation fields, definitions, and usage instructions.

---

## 1. Exploration

Exploration refers to the process of collecting the datapoints. We distinguish between two types of exploration: *Strategic* and *Non-strategic*.

### Annotated Fields:
- **Strategic exploration**  
  *Datapoint differs from one of the previously collected datapoint in only one variable.*
  - (A=1, B=2, C=3) -> (A=1, B=2, C=4)

- **Non-strategic exploration**  
  *Datapoint differs from all previous datapoints in multiple variables* (**random exploration**) *or it is the same already collected datapoint* (**repeated exploration**).

---

## 2. Reasoning

Reasoning refers to identifying the cause effect of the change. We define following types:

### Annotated Fields:
- **Verbalizing**  
  *Only describing the change in the observation.*
  - Variable D became 5.

- **Qualitative change**  
  *Identifying the effect in terms of direction.*
  - Variable D decreased.

- **Quantitative change**  
  *Categorization of the reasoning used (e.g., deductive, abductive, analogical).*
  - Variable D decreased by half.
  - Variable D went from 5 to 10.

We annotate reasoning for input and output separately and define following cause effect types:
### Metrics:
- **Verbalizing (Statement)**  
  **
  - todo

- **Qualitative effect**  
  **
  - todo

- **Quantitative effect**  
  **
  - todo

---

## 3. Verification

Verification refers to ability to engage and/or verify hypotheses.

### Annotated Fields:
- **Verification thought**  
  *The reasoning trace contains verification step.*

- **Computational verification**  
  *The verification is calculated numerically step by step.*

- **Correct computation**  
  *The numerical computation is correct.*

- **Accepted hypohtesis**  
  *The agent states it accepts hypothesis or concludes the hypothesis is supported with the observation following the verification.*

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

