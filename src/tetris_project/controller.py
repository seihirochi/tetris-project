from tetris_gym import Action
from abc import ABC, abstractmethod
from typing import Mapping


class Controller(ABC):
    def __init__(self, actions: set[Action]) -> None:
        self.actions = actions

    @abstractmethod
    def get_action(self) -> Action:
        pass


class HumanController(Controller):
    def __init__(self, actions: set[Action], input_map: Mapping[str, Action]) -> None:
        super().__init__(actions)
        self.action_map = {action.id: action for action in actions}
        self.input_map = input_map

    def get_action(self) -> Action:
        while True:
            try:
                action_input = input("Enter action: ")
                action = self.input_map[action_input]
                return self.action_map[action.id]
            except KeyError:
                print("Invalid action")
