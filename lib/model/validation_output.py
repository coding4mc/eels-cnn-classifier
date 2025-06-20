from typing import Dict, Set
import dataclasses


from .accuracy import Accuracy


@dataclasses.dataclass(frozen=True)
class ValidationOutput:
    """
    A dataclass to represent the output of a validation process.

    Attributes:
        loss (float): The loss value.
        correct_count (int): The number of correct predictions.
        total_count (int): The total number of predictions.
    """
    loss: float
    overall_accuracy: Accuracy
    accuracy_per_label: Dict[int, Accuracy]

    @property
    def validated_labels(self) -> Set[int]:
        """ All labels part of the validation. """
        return set(self.accuracy_per_label.keys())

    def merge_with(self, other: "ValidationOutput") -> "ValidationOutput":
        """
        Merge this ValidationOutput with another ValidationOutput.

        Args:
            other (ValidationOutput): The other ValidationOutput to merge with.

        Returns:
            ValidationOutput: A new ValidationOutput instance representing the merged results.
        """
        labels = self.validated_labels.union(other.validated_labels)
        return ValidationOutput(
            loss=(self.loss + other.loss) / 2,
            overall_accuracy=self.overall_accuracy.merge_with(other.overall_accuracy),
            accuracy_per_label={
                label: self.accuracy_per_label[label].merge_with(other.accuracy_per_label[label])
                for label in labels
            }
        )

    def __str__(self) -> str:
        output = "[ValidationOutput]:"
        output += f"\n- [loss]: {self.loss:.4f}"
        output += f"\n- [overall]: {self.overall_accuracy}"
        for label, label_accuracy in self.accuracy_per_label.items():
            output += f"\n- [accuracy per label]:"
            output += f"\n    - [{label}]: {label_accuracy}"

        return output

    def __repr__(self):
        return str(self)


