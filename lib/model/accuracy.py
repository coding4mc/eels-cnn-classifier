from dataclasses import dataclass


@dataclass(frozen=True)
class Accuracy:
    correct_count: int
    total_count: int
    
    @property
    def accuracy(self) -> float:
        return self.correct_count / self.total_count
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return ", ".join([
            f"accuracy: {self.accuracy:.4f}",
            f"correct_count: {self.correct_count}",
            f"total_count: {self.total_count}"
        ])
    
    def merge_with(self, other: "Accuracy") -> "Accuracy":
        return Accuracy(
            correct_count=self.correct_count + other.correct_count,
            total_count=self.total_count + other.total_count,
        )

    