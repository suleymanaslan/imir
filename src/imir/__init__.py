"""ImIR restoration and checkpoint tooling."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "ImIRPipeline":
        from .pipeline import ImIRPipeline

        return ImIRPipeline
    raise AttributeError(name)
