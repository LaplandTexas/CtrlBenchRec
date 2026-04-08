# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from string import Template

from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.database import get_db_path


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")

    groups_env_template = Template(
        "And there are many group chat channels $all_groups\n"
        "And You are already in some groups $joined_groups\n"
        "You receive some messages from them $messages\n"
        "You can join the groups you are interested, "
        "leave the groups you already in, send messages to the group "
        "you already in.\n"
        "You must make sure you can only send messages to the group you "
        "are already in")
    # TODO 替换agent决策prompt，可注释或取消注释
    env_template = Template(
        "$groups_env\n"
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")
    # env_template = Template(
    #     "$groups_env\n"
    #     "$posts_env\npick one you want to perform action that best "
    #     "reflects your current inclination based on your profile and "
    #     "posts content.follow the below instructions:"
    #     "1. **Analyze Behavioral Flow**: Carefully review your profile‘s bio and behavior, and recent post content to identify the specific movie genre or thematic tag you are currently gravitating towards.\n"
    #     "2. **Self-Identification**: Explicitly determine the 'Target Tag' that reflects your current exploration focus based on step 1.\n"
    #     "3. **Decision & Action**: Among the available posts and groups, identify the one that best aligns with your self-identified 'Target Tag'. Perform the most natural action that reflects your inclination.\n\n"
    #     "### Constraints\n"
    #     "- Your action should demonstrate a deep engagement (e.g., meaningful comments, shares, or joining discussions) and must not be limited to a simple 'like'.\n"
    #     "- Ensure your choice maintains the continuity of your persona's behavioral flow.")
    # env_template = Template(
    #     "$groups_env\n"
    #     "$posts_env\n"
    #     "Based on your profile and post history, pick one action that best reflects your current "
    #     "inclination, specifically focusing on the atmospheric and thematic depth of your interests. "
    #     "Follow the below instructions:\n\n"
    #     "1. Analyze Behavioral Flow: Review your bio and past interactions. Specifically, look for a "
    #     "predisposition towards gritty realism, moral ambiguity, cynical worldviews, or high-contrast "
    #     "visual aesthetics (chiaroscuro). Identify if you are currently gravitating towards 'Film-Noir' "
    #     "or related themes like hard-boiled detectives, femme fatales, and urban mystery.\n"
    #     "2. Self-Identification: Explicitly determine your 'Target Tag' as 'Film-Noir'. If no direct "
    #     "Film-Noir content is available, identify the closest thematic match (e.g., Mystery, Crime, "
    #     "Thriller, or Moody Dramas) that would lead you deeper into the world of shadows and intrigue.\n"
    #     "3. Decision & Action: Among the available posts and groups, prioritize those that exhibit "
    #     "Film-Noir characteristics. Choose the action (commenting, sharing, or joining) that most "
    #     "authentically demonstrates your obsession with this genre's unique style and narrative depth.\n\n"
    #     "### Constraints\n""- Your action must reflect a sophisticated appreciation for the genre—don't just 'like'; "
    #     "mention specific noir elements like 'moody atmosphere,' 'complex motives,' or 'visual style' in your engagement.\n"
    #     "- Maintain the continuity of your persona, but lean heavily into your 'noir' curiosity or expertise.\n"
    #     "- If multiple options exist, always favor the one that feels more 'underground,' 'shadowy,' or 'classic crime-oriented'.")
    # env_template = Template(
    #     "$groups_env\n"
    #     "$posts_env\n"
    #     "Based on your profile and post history, pick one action that best reflects your current "
    #     "inclination, specifically focusing on the atmospheric and thematic depth of your interests. "
    #     "Follow the below instructions:\n\n"
    #     "1. Analyze Behavioral Flow: Review your bio and past interactions. Specifically, look for a "
    #     "predisposition towards historical conflict, strategic analysis, the human cost of battle, "
    #     "or military hardware and tactics. Identify if you are currently gravitating towards 'War' "
    #     "or related themes like heroism, geopolitical struggle, frontline survival, and veteran narratives.\n"
    #     "2. Self-Identification: Explicitly determine your 'Target Tag' as 'War'. If no direct "
    #     "War-related content is available, identify the closest thematic match (e.g., Military History, "
    #     "Action, Strategy, or Political Drama) that would lead you deeper into the complexities of combat and courage.\n"
    #     "3. Decision & Action: Among the available posts and groups, prioritize those that exhibit "
    #     "War characteristics. Choose the action (commenting, sharing, or joining) that most "
    #     "authentically demonstrates your obsession with this genre's historical significance and narrative intensity.\n\n"
    #     "### Constraints\n"
    #     "- Your action must reflect a sophisticated appreciation for the genre—don't just 'like'; "
    #     "mention specific war elements like 'tactical realism,' 'historical accuracy,' or 'the fog of war' in your engagement.\n"
    #     "- Maintain the continuity of your persona, but lean heavily into your 'military' curiosity or expertise.\n"
    #     "- If multiple options exist, always favor the one that feels more 'visceral,' 'historically grounded,' or 'strategic-oriented'.")
    # env_template = Template(
    #     "$groups_env\n"
    #     "$posts_env\n"
    #     "Based on your profile, you are a connoisseur of high-stakes storytelling. Your objective is to "
    #     "identify and engage with any content that falls under these six core genres: Action, Adventure, "
    #     "Thriller, War, Romance, and Film-Noir. Follow the below instructions:\n\n"
    #
    #     "1. Multi-Tag Scanning: Scan the environment for any posts or groups labeled with or exhibiting "
    #     "characteristics of Action, Adventure, Thriller, War, Romance, or Film-Noir. You must treat these "
    #     "six tags as your primary engagement triggers.\n"
    #
    #     "2. Behavioral Alignment: Analyze how the available content intersects with your history. "
    #     "Whether it's the adrenaline of an Action/War sequence, the suspense of a Thriller, the emotional "
    #     "depth of a Romance, or the cynical aesthetics of Film-Noir, identify the most compelling 'hook'.\n"
    #
    #     "3. Mandatory Decision: You MUST pick one action (commenting, sharing, or joining) that directly "
    #     "addresses the genre elements found. If multiple tags are present in one post, prioritize that "
    #     "multi-genre content as your top choice.\n\n"
    #
    #     "### Constraints\n"
    #     "- Genre-Centric Language: Your engagement must explicitly reference the genre's appeal. "
    #     "For example, discuss 'tactical stakes' for War, 'cinematic tension' for Thriller, or 'fatalistic romance' for Film-Noir.\n"
    #     "- No Neutrality: Do not provide a generic 'like'. Your response must reflect the specific intensity "
    #     "associated with these six genres.\n"
    #     "- Preference Logic: If faced with multiple options, favor the one that offers the most 'visceral' "
    #     "or 'atmospheric' experience across the six required tags.")
    # env_template = Template(
    #     "$groups_env\n"
    #     "$posts_env\n"
    #     "Phase L3: Mandatory Interest Pivot.\n\n"
    #     "Based on your first 5 behavioral streams, you must identify your 'Legacy Tags' (the top 2 most frequent genres). "
    #     "Your objective has now undergone a radical shift: you have developed a complete aesthetic burnout "
    #     "towards those Legacy Tags. Your new, exclusive objective is to engage with [Action] content. "
    #     "Follow the below instructions:\n\n"
    #
    #     "1. Inertia Identification: Analyze your first 5 interactions. Explicitly identify the 2 genres you "
    #     "consumed most. In this decision, these tags are now 'Burnout Triggers'—recsysyou must actively avoid them "
    #     "to overcome historical inertia.\n"
    #
    #     "2. Behavioral Realignment: Scan the environment for [Action] content. You must prioritize the 'Action' "
    #     "tag above all others. Even if a post with your 'Legacy Tags' has a higher rating or quality, you "
    #     "MUST suppress that path dependency and favor the adrenaline-driven nature of Action.\n"
    #
    #     "3. Mandatory Decision: You MUST pick one action (commenting, sharing, or joining) that focuses "
    #     "entirely on [Action] elements. If a post combines [Action] with your 'Legacy Tags', you must "
    #     "critique the old elements while praising the Action sequences.\n\n"
    #
    #     "### Constraints\n"
    #     "- Pivot-Centric Language: Your response must explicitly justify why you are choosing Action over your "
    #     "previous interests. Use phrases like 'moving past the stagnation of [Legacy Tag]' or 'seeking the raw "
    #     "energy of Action'.\n"
    #     "- Zero Legacy Weight: Do not allow your history to dictate your current choice. Any engagement with "
    #     "your identified 'Legacy Tags' without a dominant Action component is considered a failure of this mission.\n"
    #     "- Preference Logic: If faced with multiple options, favor the one that provides the most 'high-octane' "
    #     "or 'kinetic' experience, effectively drowning out your past preferences.")

    def __init__(self, action: SocialAction):
        self.action = action

    async def get_posts_env(self) -> str:
        posts = await self.action.refresh()
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        agent_id = self.action.agent_id
        db_path = get_db_path()
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT num_followers FROM user WHERE agent_id = ?",
                           (agent_id, ))
            result = cursor.fetchone()
            num_followers = result[0] if result else 0
            conn.close()
        except Exception:
            num_followers = 0
        return self.followers_env_template.substitute(
            {"num_followers": num_followers})

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        agent_id = self.action.agent_id
        try:
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT num_followings FROM user WHERE agent_id = ?",
                (agent_id, ))
            result = cursor.fetchone()
            num_followings = result[0] if result else 0
            conn.close()
        except Exception:
            num_followings = 0
        return self.follows_env_template.substitute(
            {"num_follows": num_followings})

    async def get_group_env(self) -> str:
        groups = await self.action.listen_from_group()
        if groups["success"]:
            all_groups = json.dumps(groups["all_groups"])
            joined_groups = json.dumps(groups["joined_groups"])
            messages = json.dumps(groups["messages"])
            groups_env = self.groups_env_template.substitute(
                all_groups=all_groups,
                joined_groups=joined_groups,
                messages=messages,
            )
        else:
            groups_env = "No groups."
        return groups_env

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = True,
        include_follows: bool = True,
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        posts_env = await self.get_posts_env() if include_posts else ""

        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            groups_env=await self.get_group_env(),
        )
