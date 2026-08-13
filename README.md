# Collide, Orbit, or Escape

MAT244 Final project - a computational study of attraction laws from pursuit to gravity 

Team members:
Ray Fang
Taeho Kim 
Hejie Guan
Aurélien Fang

Two bodies pulled toward each other can settle into an orbit, escape, or collide.
This project asks what structural property of the attraction law decides which.
The answer we test is that it is not the strength of the attraction but the order at
which the law constrains the motion: **velocity-level pursuit forces `L = 0` and
captures in finite time, while force-level gravity leaves `L` free and therefore
orbits.** Both are then embedded in the family `V(ρ) = -α ρ^-s` to find where the
transitions actually sit.

## Running it

```bash
pip install -r requirements.txt
python -m pytest

python experiments/exp0_solver_validation.py   
python experiments/exp1_dichotomy.py           
python experiments/exp2_outcome_map.py         
python experiments/exp3_bertrand.py            
```