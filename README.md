# CrewForge

CrewForge is a task-agnostic multi-agent LLM system that allows dynamic creation of agent teams based on a given task. A Meta Agent oversees this process, allowing for mid-task reflection during execution.

---

## Project Structure

### Layout of Repository

- All backend code is in the `src/` folder

### Description of Modules

- `run_gcc_experiments.py` — the main entrypoint to running experiments. The command-line arguments can be used to specify the experiment setup (alternatively, this can be done by editing the constants at the top of the file, such as `MODELS`)
- `runner.py` — used to run a single run within a batch of experiments
- `environment.py` — the ABC for any environment. To add an environment to CrewForge, it must be compatible with the layout of this file.
- `system.py` — the ABC for LLM integration into environments. This is agnostic to the environment, all information is passed via dataclasses defined in the `environment.py`. The two systems implemented are the `single_system.py`, which is the single LLM baseline, and the `multi_system.py`, which is the CrewForge system.
- `chess_env.py` — the chess environment, implements the `environment.py` ABC
- `memory_module.py` — implements the memory registry for the CrewForge system.
- all other files are helper/auxiliary modules for the above.

## Environment Setup

### Prerequisites

- Python 3.12+ & packages (see requirements.txt)
- [Stockfish](https://stockfishchess.org/) binary (for chess environment)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/CrewForge/CrewForge.git
   cd CrewForge/
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```


4. **Download the Stockfish Binary**

    ```bash
    wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar
    tar -xvf stockfish-ubuntu-x86-64-avx2.tar
    ```


5. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   STOCKFISH_PATH=/path/to/stockfish/binary
   GCLOUD_PROJECT=/path/to/your/gcloud project
   ```


### Running the Application

See `run_gcc_experiments.py`. The `HARDNESS_LEVELS` dict controls the difficulty tiers to use, the `MODES` dict controls which system to use, and the `MODELS` dict controls the LLM models to use.