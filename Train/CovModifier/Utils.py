from typing import TypeVar, Generic, Literal, Any

T = TypeVar("T")

class CircularBuffer(Generic[T]):
    def __init__(self, window_size: int):
        window_size += 1
        self.size  = window_size
        self.buffer: list[None | T] = [None for _ in range(window_size)]
        self.start : int = 0
        self.end   : int = 0
        self.length: int = 0
    
    def push(self, v: T):
        self.length += 1
        self.buffer[self.end] = v
        self.end = (self.end + 1) % self.size
        if self.end == self.start: self.start = (self.start + 1) % self.size

    def __repr__(self) -> str:
        return f"{self.buffer=}, {self.vec=}, {self.start=}, {self.end=}"

    @property
    def vec(self) -> list[T]:
        if self.end > self.start: return self.buffer[self.start:self.end]   #type: ignore
        else: return self.buffer[self.start:] + self.buffer[:self.end]      #type: ignore


class TrainInstabilityException(Exception):
    def __init__(self, msg, *args: object) -> None:
        super().__init__(*args)
        self.msg = msg


class DataLogService:
    T_Source = Literal["wandb", "rerun", "stdout"]
    
    class WandbProvider:
        def __init__(self) -> None:
            import wandb
            self.module = wandb
        
        def log(self, **kwargs):
            self.module.log(kwargs)
    
    class RerunProvider:
        def __init__(self) -> None:
            import rerun
            self.module = rerun
            self.step   = 0
            self.logged = set()
        
        def log(self, **kwargs):
            self.module.set_time_sequence('optim_step', self.step)
            self.step += 1
            for key, value in kwargs.items():
                if key not in self.logged:
                    self.module.log(f"/Log/{key}", self.module.SeriesLine(width=2, name=key))
                    self.logged.add(key)
                
                self.module.log(
                    f"/Log/{key}", self.module.Scalar(scalar=value)
                )
    
    class StdoutProvider:
        def log(self, **kwargs):
            for key, value in kwargs.items(): print(f"{key}={value}")
    
    def __init__(self, to: T_Source, project_name: str) -> None:
        self.context: DataLogService.WandbProvider | DataLogService.RerunProvider | DataLogService.StdoutProvider
        
        match to:
            case "wandb":
                import wandb
                wandb.init(project=project_name)
                self.context = DataLogService.WandbProvider()
            case "rerun":
                import rerun
                rerun.init(application_id=project_name, spawn=True)
                rerun.connect_tcp()
                self.context = DataLogService.RerunProvider()
            case "stdout":
                self.context = DataLogService.StdoutProvider()
    
    def write_entries(self, **kwargs: float):
        self.context.log(**kwargs)


class ScalarScheduler:
    def __init__(self, steps: list[int], values: list[int]) -> None:
        """
        The scheduler will return values[i] when step is smaller than step[i].
        The values[-1] will be used when step is larger or equal to step[-1].
        """
        assert len(values) == len(steps) + 1
        self.cur_step  = 0
        self.cur_stage = 0
        self.values    = values
        self.steps     = steps
    
    def get_value(self) -> int:
        return self.values[self.cur_stage]
    
    def step(self):
        self.cur_step += 1
        
        while self.cur_stage < len(self.steps) and self.cur_step >= self.steps[self.cur_stage]:
            self.cur_stage += 1
