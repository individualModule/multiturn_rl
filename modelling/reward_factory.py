from abc import ABC, abstractmethod

WORDLE_EXPLANATION_LANG = "explanation:"
WORDLE_GUESS_LANG = "guess:"
class RewardFactory:
    """
    Factory class to manage and create reward logic instances.
    """

    _reward_logic_map = {
        "wordle": WordleRewardLogic
    }

    @staticmethod
    def get_reward_logic(game_name: str) -> RewardLogic:
        """
        Get the reward logic instance based on the logic name.

        Args:
            game_name (str): The name of the reward logic.

        Returns:
            RewardLogic: An instance of the reward logic.
        """
        if game_name not in RewardFactory._reward_logic_map:
            raise ValueError(f"Unknown reward logic: {logic_name}")
        return RewardFactory._reward_logic_map[game_name]()


class RewardLogic(ABC):
    """
    Abstract base class for game-specific reward logic
    """


    @abstractmethod
    def calculate(self, pi_action: list) -> torch.Tensor:
        """
        Calculate a verifiable reward using game-specific logic for the given action.
        """
        pass

    


class WordleRewardLogic(RewardLogic):

    def calculate(self, pi_action: list) -> torch.Tensor:

        pass

    def parse_response(player: Player, response: str, words: Dict) -> Tuple[str, str]:

    
        """Parse guesser response and extract guess and explanation"""
        if not response or not response.startswith(WORDLE_EXPLANATION_LANG):
            return None
            
        response = response.strip()
        lines = response.split("\n")
        if len(lines) > 2:
            return None

        # Extract explanation and guess
        explanation_pattern = re.compile(rf"{WORDLE_EXPLANATION_LANG}([^\n]*)", re.IGNORECASE)

        content_prefix = words['guess_lang']
        if isinstance(player, WordCritic):
            content_prefix = words['agreement_lang']
        content_pattern = re.compile(rf"{content_prefix}([^\n]*)", re.IGNORECASE)

        explanation_match = explanation_pattern.search(response)
        content_match = content_pattern.findall(response)

        if len(content_match) != 1:
            raise ParseError(f"The response should contain the '{content_prefix}' keyword exactly once.",
                            key="MORE_THAN_ONE_GUESS")

        content = content_match[0].strip().lower()
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        return content, explanation
