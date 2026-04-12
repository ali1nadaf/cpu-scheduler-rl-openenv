from fastapi import FastAPI
from cpu_env import CPUSchedulerEnv

app = FastAPI()
env = CPUSchedulerEnv()

@app.post("/reset")
def reset():
    return {"state": env.reset()}

@app.post("/step")
def step(action: int):
    state, reward, done, info = env.step(action)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return {"state": env.get_state()}