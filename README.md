# Get Started...

1. previous demo moved to /_old
2. model_endpoint using Memgpt "https://inference.memgpt.ai", free
3. frontend is not so good, bearly ok

## TODO

1. Multi agent
2. Create custom tools

### Setup the Backend

Install Poetry, visit [Poetry Documentation](https://python-poetry.org/docs/).

```bash
poetry install
```

### Build the Frontend

```bash
cd frontend
npm install
npm run build
```

### Create the agent

```bash
cd backend
python create_agent.py
```

### Start the app

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug 
```

## Accessing the Frontend

The React frontend is hosted statically through the backend. You can access it by navigating to:

http://localhost:8000/