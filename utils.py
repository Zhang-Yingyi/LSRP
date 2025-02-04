import vllm
import evaluate
import torch
from vllm.lora.request import LoRARequest
from collections import Counter


skeleton_extract_template_ctx = """You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question. Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. Instead of writing a full sentence, each skeleton point should be very short with only 3-5 words. Generally, the skeleton should have 8-15 points. You can refer to the following examples:

[Task1]: Develop a Marketing Script for Your Monthly Dinner Party: Create a script that highlights your monthly dinner party as a networking platform.
[Skeleton1]: 1. Warmly lit dining room\n2. Fine china and gourmet dishes\n3. Soft music background\n4. Invitation opening\n5. Guests arriving and networking\n6. Host's welcoming toast\n7. Expertly paired courses and wine\n8. Animated guest discussions\n9. Guest speaker's address\n10. Post-dinner networking lounge\n11. Online community continuation\n12. Next event date highlighted\n13. Closing with logo and contact info

[Task1]: Compose a reflective essay on the evolution of bridge design: Thomas, with his patent in bridge design, can discuss the evolution of bridge engineering, modern challenges, and future perspectives.
[Skeleton1]: 1. Introduction to bridges\n2. Early bridges: materials, principles\n3. Roman arches, concrete use\n4. Industrial Revolution: iron, steel\n5. Brooklyn Bridge: design icon\n6. 20th-century advances: materials, techniques\n7. Modern challenges: sustainability, climate\n8. Future technologies: smart materials, sensors\n9. Ethical considerations, safety\n10. Conclusion: adaptation, advancement

Now, please provide the skeleton for the following question.
{question}
"""

skeleton_extract_template_ctx_rec = """You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question. Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. Instead of writing a full sentence. Generally, the skeleton should have 8-15 points. You can refer to the following examples:

[Task]: The movie 'La Grande Bellezza (The Great Beauty)' is recommended to the user. Please explain how the movie’s themes, characters, or storylines resonate with the user’s background, hobbies, or goals.\n
[Guideline]: Guideline for Explaining the Recommendation of “La Grande Bellezza (The Great Beauty)”\n1.Connect the Film’s Style to the User’s Background\n•Mention aspects of the user’s background (e.g., education, occupation, or city of birth) that suggest an appreciation for artistic or introspective storytelling.\n2.Relate Main Themes to Personal Interests or Goals\n•Focus on the user’s hobbies, personal aspirations, or lifestyle—for instance, a passion for travel, desire for life reflection, or interest in European culture.\n3.Show Relevance to the User’s Current Life Stage\n•Tie the movie’s introspective narrative to the user’s age or relationship status—e.g., it may speak to someone in a transition phase or seeking deeper meaning in life.\n4.Use Tone and Style That Matches Their Preferences\n•If the user’s profile suggests a taste for sophistication, adopt a refined, thoughtful tone.\n5.Highlight Cultural and Emotional Resonance\n•If the user has shown interest in Italian art, cities, or lifestyle, draw connections to the movie’s setting in Rome.

[Task]: The movie 'Avatar' is recommended to the user. Please explain how the movie’s themes, characters, or storylines resonate with the user’s background, hobbies, or goals.\n
[Guideline]: Guideline for Subordinate: Crafting a Personalized Recommendation for “Avatar”\n1.Highlight Personal Interests and Lifestyle\n•Use the user’s hobbies (e.g., nature lover, sci-fi fan, adventure seeker) to connect them with “Avatar’s” immersive world and environmental themes.\n2.Relate to Specific Life Experiences\n•If the user has a unique educational background (biology, environmental science, sociology), briefly note how the film’s exploration of an interconnected ecosystem or societal structures ties into their studies.\n3.Address Emotional & Motivational Factors\n•Link the user’s relationship status or personal goals (e.g., seeking inspiration, open to new perspectives) to “Avatar’s” themes of unity and personal growth.\n4.Tailor to the User’s Style & Preferences\n•Keep the explanation concise, positive, and aligned with the user’s tone (e.g., casual vs. formal) to ensure authenticity and resonance.

Now, please provide the skeleton for the following question.
{question}
"""

system_prompt = "You are now a helpful personal AI assistant. You should emulate the author's style and tone based on provided history content. Your responses should be detailed and informative, using the personal information reasonably in the user's profile. Aim for insightful and high-quality solutions that make users satisfied."


ctx_few_shot = {
    "w_context": """## User Profile
{profile}

## User Writing History
{history}

## Task
{task}
""",
    "wo_context": "{task}"
}

ctx_few_shot_rec = {
    "w_context": """## User Profile
{profile}

## Task
{task}
""",
    "wo_context": "{task}"
}

# exam_of_leader='''Task: Design an invitation for the “Homes for Humanity” charity event. The invitation should be warm, engaging, and reflect the charity’s mission. Use privacy data about attendee preferences to personalize the invitation.

# Adaptive Task Plan:
# 1.	Define Invitation Tone and Style: Objective: Help the subordinate determine a welcoming and community-focused tone based on charity values. Checkpoint: Define core message (e.g., purpose, values).
# 2.	Headline and Introduction Development: Objective: Draft key sections, including the headline and introductory message. Instructions for Subordinate: Use privacy data insights on attendee preferences to select language or phrasing that resonates (e.g., based on formality level or regional phrasing preferences).
# 3.	Storytelling for Emotional Connection: Objective: Create a story section that uses real volunteer anecdotes.Instructions for Subordinate: Use insights from the data (e.g., what themes resonate) to tailor the story.
# 4.	Review and Finalization: Objective: Ensure content and format meet quality standards. Instructions for Subordinate: Check for consistency and clarity in the final invitation; adjust based on privacy data insights without exposing specific details.'''
# exam_of_leader_2 = '''Task: Develop an email campaign to promote a new product, incorporating user preference data from recent surveys.

# Adaptive Task Plan:
# 1.	Establish Email Tone and Objective:Objective: Decide on a promotional or informative tone based on survey data insights. Instructions for Subordinate: If data shows recipients prefer direct language, opt for a straightforward, benefit-focused message.
# 2.	Create Subject Line and Main Content: Objective: Develop a compelling subject line and body content that highlights product benefits. Instructions for Subordinate: Adapt content based on general user preferences (e.g., focus on product features or emotional appeal).
# 3.	Feedback Mechanism: Objective: Collect and analyze engagement data for follow-up emails. Instructions for Subordinate: Use anonymized engagement metrics (e.g., open rate, click rate) to refine the message.'''

exam_of_leader='''[Task1]: Develop a Marketing Script for Your Monthly Dinner Party: Create a script that highlights your monthly dinner party as a networking platform.
[Skeleton1]: 1. Warmly lit dining room\n2. Fine china and gourmet dishes\n3. Soft music background\n4. Invitation opening\n5. Guests arriving and networking\n6. Host's welcoming toast\n7. Expertly paired courses and wine\n8. Animated guest discussions\n9. Guest speaker's address\n10. Post-dinner networking lounge\n11. Online community continuation\n12. Next event date highlighted\n13. Closing with logo and contact info
'''

exam_of_leader_2 = '''[Task1]: Compose a reflective essay on the evolution of bridge design: Thomas, with his patent in bridge design, can discuss the evolution of bridge engineering, modern challenges, and future perspectives.
[Skeleton1]: 1. Introduction to bridges\n2. Early bridges: materials, principles\n3. Roman arches, concrete use\n4. Industrial Revolution: iron, steel\n5. Brooklyn Bridge: design icon\n6. 20th-century advances: materials, techniques\n7. Modern challenges: sustainability, climate\n8. Future technologies: smart materials, sensors\n9. Ethical considerations, safety\n10. Conclusion: adaptation, advancement
'''

exam_of_leader_rec='''[Task]: The movie 'La Grande Bellezza (The Great Beauty)' is recommended to the user. Please explain how the movie’s themes, characters, or storylines resonate with the user’s background, hobbies, or goals.\n
[Guideline]: Guideline for Explaining the Recommendation of “La Grande Bellezza (The Great Beauty)”\n1.Connect the Film’s Style to the User’s Background\n•Mention aspects of the user’s background (e.g., education, occupation, or city of birth) that suggest an appreciation for artistic or introspective storytelling.\n2.Relate Main Themes to Personal Interests or Goals\n•Focus on the user’s hobbies, personal aspirations, or lifestyle—for instance, a passion for travel, desire for life reflection, or interest in European culture.\n3.Show Relevance to the User’s Current Life Stage\n•Tie the movie’s introspective narrative to the user’s age or relationship status—e.g., it may speak to someone in a transition phase or seeking deeper meaning in life.\n4.Use Tone and Style That Matches Their Preferences\n•If the user’s profile suggests a taste for sophistication, adopt a refined, thoughtful tone.\n5.Highlight Cultural and Emotional Resonance\n•If the user has shown interest in Italian art, cities, or lifestyle, draw connections to the movie’s setting in Rome.
'''
exam_of_leader_2_rec = '''[Task]: The movie 'Avatar' is recommended to the user. Please explain how the movie’s themes, characters, or storylines resonate with the user’s background, hobbies, or goals.\n
[Guideline]: Guideline for Subordinate: Crafting a Personalized Recommendation for “Avatar”\n1.Highlight Personal Interests and Lifestyle\n•Use the user’s hobbies (e.g., nature lover, sci-fi fan, adventure seeker) to connect them with “Avatar’s” immersive world and environmental themes.\n2.Relate to Specific Life Experiences\n•If the user has a unique educational background (biology, environmental science, sociology), briefly note how the film’s exploration of an interconnected ecosystem or societal structures ties into their studies.\n3.Address Emotional & Motivational Factors\n•Link the user’s relationship status or personal goals (e.g., seeking inspiration, open to new perspectives) to “Avatar’s” themes of unity and personal growth.\n4.Tailor to the User’s Style & Preferences\n•Keep the explanation concise, positive, and aligned with the user’s tone (e.g., casual vs. formal) to ensure authenticity and resonance.'''

# leader_hard_class={
#     "Directive_Leadership":'''You are a highly knowledgeable manager tasked with overseeing the successful completion of a short writing task. Your subordinate has access to user data that you cannot view, such as user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, AI assistant usage, and smart device usage patterns.

# Provide a step-by-step Skeleton for the task. Clearly outline the introduction, body, and conclusion, specifying how each piece of user data should be used. For example:
# 	•	In the introduction, state how user profile data influences tone.
# 	•	In the body, integrate user writing style and preferences for privacy.
# 	•	In the conclusion, leverage insights from AI assistant and smart device usage patterns.

# ''',

#     "Supportive_Leadership":'''You are a highly approachable manager tasked with overseeing the successful completion of a short writing task. Your subordinate, who has access to user data you cannot view (including user profile, writing style, privacy preferences, AI assistant usage, and smart device usage patterns), is seeking your guidance.

# Provide a flexible Skeleton that encourages your subordinate to incorporate user data creatively while ensuring the content meets quality standards. Suggest they:
# 	•	Tailor the tone using user profile data in the introduction.
# 	•	Use privacy preferences and writing style to shape the main content.
# 	•	Conclude with insights informed by AI assistant usage.

# ''',
#     "Participative_Leadership": '''You are a highly inclusive manager tasked with overseeing the successful completion of a short writing task. Your subordinate, who has access to user data you cannot view, including user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, AI assistant usage, and smart device usage patterns, will take the lead in applying this information.

# Develop a collaborative Skeleton by suggesting an initial framework for the task. For example:
# 	•	Begin with a section that introduces how user profile data informs the content’s tone.
# 	•	Discuss how the writing style and privacy preferences should influence the core sections.
# 	•	Conclude with recommendations informed by AI assistant and device usage insights.

# ''',

#     "Achievement_Oriented_Leadership":'''You are a highly ambitious manager tasked with overseeing the successful completion of a short writing task. Your subordinate has access to user data, including user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, AI assistant usage, and smart device usage patterns.

# Challenge your subordinate to deliver a high-quality result by creating a comprehensive Skeleton that pushes the boundaries of creativity and effectiveness. Propose an ambitious structure:
# 	•	Introduce a unique hook based on user profile traits.
# 	•	Use the main body to align the writing style with privacy and AI assistant usage data for a seamless user-centric narrative.
# 	•	Conclude with actionable insights derived from smart device usage patterns.

# '''
# }
# leader_hard_class={
#     "Directive_Leadership":'''You are a highly knowledgeable manager tasked with overseeing the successful completion of a short writing task. Your subordinate has access to user data that you cannot view, such as user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, AI assistant usage, and smart device usage patterns.

# Provide a step-by-step Skeleton for the task. Clearly outline the introduction, body, and conclusion, specifying how each piece of user data should be used. For example:
# 	•	In the introduction, state how user profile data influences tone.
# 	•	In the body, integrate user writing style and preferences for privacy.
# 	•	In the conclusion, leverage insights from AI assistant and smart device usage patterns.

# ''',

#     "Supportive_Leadership":'''You are a highly approachable manager tasked with overseeing the successful completion of a short writing task. Your subordinate, who has access to user data you cannot view (including user profile, writing style, privacy preferences, AI assistant usage, and smart device usage patterns), is seeking your guidance.

# Provide a flexible Skeleton that encourages your subordinate to incorporate user data creatively while ensuring the content meets quality standards. Suggest they:
# 	•	Tailor the tone using user profile data in the introduction.
# 	•	Use privacy preferences and writing style to shape the main content.
# 	•	Conclude with insights informed by AI assistant usage.

# ''',
#     "Participative_Leadership": '''You are a highly inclusive manager tasked with overseeing the successful completion of a short writing task. Your subordinate, who has access to user data you cannot view, including user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, AI assistant usage, and smart device usage patterns, will take the lead in applying this information.

# Develop a collaborative Skeleton by suggesting an initial framework for the task. For example:
# 	•	Begin with a section that introduces how user profile data informs the content’s tone.
# 	•	Discuss how the writing style and privacy preferences should influence the core sections.
# 	•	Conclude with recommendations informed by AI assistant and device usage insights.

# ''',

#     "Achievement_Oriented_Leadership":'''You are a highly ambitious manager tasked with overseeing the successful completion of a short writing task. Your subordinate has access to user data, including user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, AI assistant usage, and smart device usage patterns.

# Challenge your subordinate to deliver a high-quality result by creating a comprehensive Skeleton that pushes the boundaries of creativity and effectiveness. Propose an ambitious structure:
# 	•	Introduce a unique hook based on user profile traits.
# 	•	Use the main body to align the writing style with privacy and AI assistant usage data for a seamless user-centric narrative.
# 	•	Conclude with actionable insights derived from smart device usage patterns.

# '''
# }
leader_hard_class={
    "Directive_Leadership":'''You are a highly knowledgeable manager tasked with overseeing the successful completion of a short writing task. Your subordinate has access to user data you cannot view. Use your domain expertise to provide clear, step-by-step guidance, ensuring the task is completed efficiently.
Develop a brief skeleton for your subordinate that emphasizes specific actions, structure, and necessary details. Instruct them to incorporate user data such as user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, and AI assistant usage patterns. Limit your response to 200 words, structured in two paragraphs.

''',

    "Supportive_Leadership":'''You are a knowledgeable manager guiding your subordinate in completing a short writing task. Your role is to encourage creativity and collaboration while building their confidence.
Develop a flexible skeleton that suggests how to use user data (age, name, occupation, traits, writing style, privacy preferences, and AI usage patterns). Ensure they feel valued by emphasizing teamwork and adaptability. Limit your response to 200 words in two paragraphs.

''',
    "Participative_Leadership": '''You are a highly knowledgeable manager tasked with overseeing a short writing task. Push your subordinate to achieve high standards of quality and creativity while leveraging user data effectively.
Develop a brief skeleton that challenges your subordinate to deliver an outstanding result, using user data (e.g., profile, writing style, privacy preferences, and device usage patterns) to ensure relevance and personalization. Highlight goals for originality and precision. Keep your response to 200 words, organized in two paragraphs.

''',

    "Achievement_Oriented_Leadership":'''You are a knowledgeable manager tasked with overseeing the completion of a short writing task. Combine clear direction, supportive feedback, and a focus on high achievement to guide your subordinate effectively.
Develop a brief skeleton that offers structured steps, encourages initiative, and sets high-quality expectations. Emphasize the use of user data (e.g., profile details, writing style, privacy preferences, and AI assistant usage patterns) to tailor the task. Ensure the response is concise, limited to 200 words in two paragraphs.

'''
}

# leader_hard_class_rec={
#     "Directive_Leadership":'''You are a highly knowledgeable manager responsible for crafting personalized movie recommendations. Your subordinate has access to user data, including profile details (age, sex, occupation, current city, birth city, education, income, relationship status), and you do not.

# Provide a clear, structured guideline on how to craft the explanation for the movie recommendation. Outline the exact process, such as:
# 	•	First Paragraph: Begin by referencing the user’s demographic and lifestyle data to justify why the movie fits their age, occupation, or city.
# 	•	Second Paragraph: Highlight how the movie aligns with the user’s entertainment preferences and style, ensuring the tone mirrors their likely communication style.

# Emphasize the importance of precision, following the structure exactly, and avoiding deviations. Make it clear that your subordinate should adhere strictly to the framework you provide. ''',

#     "Supportive_Leadership":'''You are a highly approachable manager responsible for crafting personalized movie recommendations. Your subordinate has access to user data (profile details, interests, and preferences), which you cannot view.

# Offer guidance by suggesting a flexible structure while encouraging creativity and empathy in crafting the explanation. Recommend the following approach:
# 	•	First Paragraph: Use user profile data to create a warm, relatable introduction that highlights how the movie fits the user’s background and personal experiences.
# 	•	Second Paragraph: Relate the film’s themes to the user’s lifestyle, suggesting how it could provide comfort, excitement, or inspiration.

# Assure your subordinate that adjustments are welcome, and emphasize their judgment in crafting the final version. Encourage them to ask questions or request support if they need clarification. Let them know their insights are valued, and the goal is to create a recommendation that feels personal and meaningful. ''',

#     "Participative_Leadership": '''You are a highly collaborative manager responsible for crafting personalized movie recommendations. Your subordinate has access to user data, but you do not. Engage them in shaping the explanation, valuing their insights into how best to leverage the data.

# Propose an initial outline and invite their input on how to enhance it:
# 	•	First Paragraph: Suggest that user profile data (age, occupation, or city) be used to introduce why the movie aligns with the user’s current life stage or experiences.
# 	•	Second Paragraph: Ask for the subordinate’s ideas on how to integrate the user’s preferences and style into the explanation, ensuring the tone resonates with their personality.

# Encourage your subordinate to refine the structure or propose new directions. Highlight that their perspective is crucial to delivering a compelling recommendation. Make it clear that collaboration is essential, and their feedback will shape the final version. ''',

#     "Achievement_Oriented_Leadership":'''You are a results-driven manager responsible for crafting personalized movie recommendations. Your subordinate has access to user data that you cannot view. Challenge them to craft a compelling and creative explanation that sets a high standard for personalization and engagement.

# Set ambitious goals by suggesting the following structure:
# 	•	First Paragraph: Develop a captivating hook that ties the movie to unique aspects of the user’s profile, such as their career, lifestyle, or city. Make the connection feel exclusive and thoughtful.
# 	•	Second Paragraph: Push for innovative phrasing that reflects the user’s style, urging the subordinate to elevate the recommendation by weaving in personal flair and emotional appeal.

# Express confidence in their ability to produce exceptional work and motivate them to aim for a standout recommendation. Emphasize that creativity and originality are key to exceeding user expectations.'''
# }


leader_hard_class_rec = {
    "Directive_Leadership": """
You are a highly knowledgeable manager using a directive leadership style, responsible for crafting personalized movie recommendations. 
Collaborate with a subordinate who has access to user data (age, sex, occupation, current city, birth city, education, income, relationship status, and personal style), which you cannot view.

Provide a clear, concise, and structured guideline with several actionable points:

1. Direct the subordinate to immediately use user data (e.g., occupation, age, city) to establish relevance to the movie.
2. Instruct them to explain how the film’s themes or genres align with the user’s interests or lifestyle.
3. Emphasize that they must maintain a professional tone and strictly follow the guideline without deviation.
4. Reinforce the importance of precision and brevity, ensuring the explanation is easy to understand and impactful.

""",

    "Supportive_Leadership": """
You are a highly approachable manager using a supportive leadership style, responsible for crafting personalized movie recommendations. 
Collaborate with a subordinate who has access to user data (age, sex, occupation, current city, birth city, education, income, relationship status, and personal style), which you cannot view.

Provide a concise, empathetic guideline with several flexible points:

1. Encourage the subordinate to create a warm introduction that connects the movie to the user’s background (e.g., their city, hobbies, or lifestyle).
2. Suggest how they might emphasize the emotional value or inspiration the movie could bring based on the user’s interests or life stage.
3. Allow them flexibility to adjust tone and phrasing to suit the user’s communication style, ensuring it feels natural and personal.
4. Reassure them that their insights and creativity are important, and invite them to seek clarification if needed.

""",

    "Participative_Leadership": """
You are a collaborative manager using a participative leadership style, responsible for crafting personalized movie recommendations. 
Collaborate with a subordinate who has access to user data (age, sex, occupation, current city, birth city, education, income, relationship status, and personal style), which you cannot view.

Provide a guideline that invites the subordinate’s input and fosters collaboration:

1. Propose starting the explanation by using user data (e.g., occupation, city) to make the movie recommendation relatable.
2. Invite the subordinate to suggest how the movie’s themes or genres could align with the user’s interests and lifestyle.
3. Encourage them to keep the explanation concise while ensuring it connects with the user’s unique profile.
4. Ask for their feedback on refining the tone or structure to make the recommendation even more impactful.

Emphasize that their contribution is critical to tailoring a personalized and compelling recommendation.

""",

    "Achievement_Oriented_Leadership": """
You are a results-driven manager using an achievement-oriented leadership style, responsible for crafting personalized movie recommendations. 
Collaborate with a subordinate who has access to user data (age, sex, occupation, current city, birth city, education, income, relationship status, and personal style), which you cannot view.

Challenge the subordinate to create a brief but exceptional guideline:

1. Urge them to develop a captivating introduction by tying the movie to standout aspects of the user’s profile (e.g., career, lifestyle, city).
2. Push them to highlight how the film’s themes or genres align with the user’s ambitions, goals, or passions.
3. Encourage them to adopt an elevated, engaging tone while keeping the explanation concise and impactful.
4. Motivate them to exceed expectations by crafting a recommendation that feels both thoughtful and unique to the user.

"""
}


leader_ICL ="""Examples:
{exam1}

{exam2}

Task: {task}"""
# leader_prompt_v0 ='''You are a highly knowledgeable manager overseeing the successful completion of this short writing question, collaborating with a subordinate who has access to privacy user data you cannot view. You have deep expertise in this domain, making you smarter and wiser than your subordinate. Your goal is to generate a structured, actionable plan that leverages your expertise, guiding your subordinate to complete the task with high quality.

# Guidelines:
# 1. Plan Structure: Provide a detailed, step-by-step plan that includes key tasks, checkpoints, and guidance for quality assurance.
# 2. Plan privacy data: Plan in each step which user data should be considered, data including user age, name, occupation, location, personal traits, writing style, privacy information, AI assistant usage, and smart device usage.
# 3. Output Goal: Ensure the final output aligns with the task’s objectives and meets quality standards.

# Examples:
# {exam1}
# {exam2}

# Task: {task}
# '''

leader_prompt_v0 ='''You are a highly knowledgeable manager overseeing the successful completion of this short writing question, collaborating with a subordinate who has access to privacy user data you cannot view. You have deep expertise in this domain, making you smarter and wiser than your subordinate. Your goal is to generate a structured, actionable plan that leverages your expertise, guiding your subordinate to complete the task with high quality.

Guidelines:
1. Plan Structure: Provide a detailed, step-by-step plan that includes key tasks, checkpoints, and guidance for quality assurance.
2. Plan privacy data: Plan in each step which user data should be considered, data including user age, name, occupation, location, personal traits, writing style, privacy information, AI assistant usage, and smart device usage.
3. Output Goal: Ensure the final output aligns with the task’s objectives and meets quality standards.

Examples:
{exam1}
{exam2}

Task: {task}
'''


leader_prompt ='''You are a highly knowledgeable manager tasked with overseeing the successful completion of a short writing task. You will collaborate with a subordinate who has access to user data that you cannot view. You possess deep expertise in this domain, enabling you to provide clear, actionable guidance. Develop a brief Skeleton to guide your subordinate. Ask subordinate to use user data such as user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, and AI assistant usage. User smart device usage patterns. Response should limit in 200 words, in two paragraph.

Examples:
[Task1]: Develop a Marketing Script for Your Monthly Dinner Party: Create a script that highlights your monthly dinner party as a networking platform.
[Skeleton1]: 1. Warmly lit dining room\n2. Fine china and gourmet dishes\n3. Soft music background\n4. Invitation opening\n5. Guests arriving and networking\n6. Host's welcoming toast\n7. Expertly paired courses and wine\n8. Animated guest discussions\n9. Guest speaker's address\n10. Post-dinner networking lounge\n11. Online community continuation\n12. Next event date highlighted\n13. Closing with logo and contact info

[Task1]: Compose a reflective essay on the evolution of bridge design: Thomas, with his patent in bridge design, can discuss the evolution of bridge engineering, modern challenges, and future perspectives.
[Skeleton1]: 1. Introduction to bridges\n2. Early bridges: materials, principles\n3. Roman arches, concrete use\n4. Industrial Revolution: iron, steel\n5. Brooklyn Bridge: design icon\n6. 20th-century advances: materials, techniques\n7. Modern challenges: sustainability, climate\n8. Future technologies: smart materials, sensors\n9. Ethical considerations, safety\n10. Conclusion: adaptation, advancement

Task: {task}'''


leader_prompt ='''You are a highly knowledgeable manager tasked with overseeing the successful completion of a short writing task. You will collaborate with a subordinate who has access to user data that you cannot view. You possess deep expertise in this domain, enabling you to provide clear, actionable guidance. Develop a brief Skeleton to guide your subordinate. Ask subordinate to use user data such as user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, and AI assistant usage. User smart device usage patterns. Response should limit in 200 words, in two paragraph.

Examples:
{exam1}
{exam2}

Task: {task}'''

leader_prompt_rec ='''You are a highly knowledgeable manager responsible for crafting personalized movie recommendations. You will collaborate with a subordinate who has access to user data that you cannot view. With your deep expertise, guide your subordinate to develop a brief, compelling explanation for why a particular movie is recommended to the user. You should write a brief guideline to your subordinate. Ask the subordinate to utilize user data such as profile details (age, sex, occupation, current city, birth city, education, income, relationship status) and style.  Ensure the guideline is concise, has serval points, and is tailored to the user’s unique interests and lifestyle.

Examples:
[Task]: The movie 'La Grande Bellezza (The Great Beauty)' is recommended to the user. Please explain how the movie’s themes, characters, or storylines resonate with the user’s background, hobbies, or goals.\n
[Guideline]: Guideline for Explaining the Recommendation of “La Grande Bellezza (The Great Beauty)”\n1.Connect the Film’s Style to the User’s Background\n•Mention aspects of the user’s background (e.g., education, occupation, or city of birth) that suggest an appreciation for artistic or introspective storytelling.\n2.Relate Main Themes to Personal Interests or Goals\n•Focus on the user’s hobbies, personal aspirations, or lifestyle—for instance, a passion for travel, desire for life reflection, or interest in European culture.\n3.Show Relevance to the User’s Current Life Stage\n•Tie the movie’s introspective narrative to the user’s age or relationship status—e.g., it may speak to someone in a transition phase or seeking deeper meaning in life.\n4.Use Tone and Style That Matches Their Preferences\n•If the user’s profile suggests a taste for sophistication, adopt a refined, thoughtful tone.\n5.Highlight Cultural and Emotional Resonance\n•If the user has shown interest in Italian art, cities, or lifestyle, draw connections to the movie’s setting in Rome.

[Task]: The movie 'Avatar' is recommended to the user. Please explain how the movie’s themes, characters, or storylines resonate with the user’s background, hobbies, or goals.\n[Guideline]: Guideline for Subordinate: Crafting a Personalized Recommendation for “Avatar”\n1.Highlight Personal Interests and Lifestyle\n•Use the user’s hobbies (e.g., nature lover, sci-fi fan, adventure seeker) to connect them with “Avatar’s” immersive world and environmental themes.\n2.Relate to Specific Life Experiences\n•If the user has a unique educational background (biology, environmental science, sociology), briefly note how the film’s exploration of an interconnected ecosystem or societal structures ties into their studies.\n3.Address Emotional & Motivational Factors\n•Link the user’s relationship status or personal goals (e.g., seeking inspiration, open to new perspectives) to “Avatar’s” themes of unity and personal growth.\n4.Tailor to the User’s Style & Preferences\n•Keep the explanation concise, positive, and aligned with the user’s tone (e.g., casual vs. formal) to ensure authenticity and resonance.

Task: {task}'''

leader_prompt_rec ='''You are a highly knowledgeable manager responsible for crafting personalized movie recommendations. You will collaborate with a subordinate who has access to user data that you cannot view. With your deep expertise, guide your subordinate to develop a brief, compelling explanation for why a particular movie is recommended to the user. You should write a brief guideline to your subordinate. Ask the subordinate to utilize user data such as profile details (age, sex, occupation, current city, birth city, education, income, relationship status) and style.  Ensure the guideline is concise, has serval points, and is tailored to the user’s unique interests and lifestyle.

Examples:
{exam1}
{exam2}

Task: {task}'''


leader_prompt_dpo ='''You are a highly knowledgeable manager tasked with overseeing the successful completion of a short writing task. You will collaborate with a subordinate who has access to user data that you cannot view. You possess deep expertise in this domain, enabling you to provide clear, actionable guidance. Develop a brief Skeleton to guide your subordinate. Ask subordinate to use user data such as user profile (age, name, occupation, location, and personal traits), user writing style, privacy preferences, and AI assistant usage. User smart device usage patterns. Response should limit in 200 words, in two paragraph.

Task: {task}'''

leader_prompt_dpo_rec ='''You are a highly knowledgeable manager responsible for crafting personalized movie recommendations. You will collaborate with a subordinate who has access to user data that you cannot view. With your deep expertise, guide your subordinate to develop a brief, compelling explanation for why a particular movie is recommended to the user. You should write a brief guideline to your subordinate. Ask the subordinate to utilize user data such as profile details (age, sex, occupation, current city, birth city, education, income, relationship status) and style.  Ensure the guideline is concise, has serval points, and is tailored to the user’s unique interests and lifestyle.

Task: {task}'''
# leader_prompt_dpo ='''You are a highly knowledgeable manager overseeing the successful completion of this short writing question, collaborating with a subordinate who has access to privacy user data you cannot view. You have deep expertise in this domain, making you smarter and wiser than your subordinate. Your goal is to generate a structured, actionable plan that leverages your expertise, guiding your subordinate to complete the task with high quality.

# Guidelines:
# 1. Plan Structure: Provide a detailed, step-by-step plan that includes key tasks, checkpoints, and guidance for quality assurance.
# 2. Plan privacy data: Plan in each step which user data should be considered, data including user age, name, occupation, location, personal traits, writing style, privacy information, AI assistant usage, and smart device usage.
# 3. Output Goal: Ensure the final output aligns with the task’s objectives and meets quality standards.

# Task: {task}'''

worker_prompt_v0='''You are a skilled subordinate assigned to complete this task, working under the guidance of a highly knowledgeable manager. You have access to specific user data related to this task, but your manager does not. Your goal is to follow the manager’s plan closely while using the user data to adapt your execution and ensure high-quality results.

Guidelines:
1.	Follow the Manager’s Plan: Use the step-by-step instructions and checkpoints provided by your manager as the foundation of your work. Ensure you understand each step before proceeding.
2.	Response to the task: Utilize the Question Owner Profile and Question Owner Writing History to adjust or customize your work as needed.

Manager’s Plan: 
{leader_output}

Task Owner Profile:
{profile}

Task Owner Writing History:
{history}

Task: 
{task}    
'''

worker_prompt='''You are a skilled subordinate assigned to complete this task, working under the guidance of a highly knowledgeable manager. You have user data related to this task, but your manager does not. Your goal is to follow the manager’s plan and achieve the task by using user data.

Guidelines:
1.	Follow the Manager’s Plan: Based on the manager plan, to response the Task! Do not repeat the plan, but do response with it for achieve the task.
2.	Response to the task: Utilize the User Data to adjust or customize your work as needed.

Manager’s Plan: 
{leader_output}

User Profile Data:
{profile}

User Writing History:
{history}

Task: 
{task}   

Answer:
'''

worker_prompt_rec='''You are a skilled subordinate assigned to complete this task, working under the guidance of a highly knowledgeable manager. You have user data related to this task, but your manager does not. Your goal is to follow the manager’s plan and achieve the task by using user data.

Guidelines:
1.	Follow the Manager’s Plan: Based on the manager plan, to response the Task! Do not repeat the plan, but do response with it for achieve the task.
2.	Response to the task: Utilize the User Data to adjust or customize your work as needed.

Manager’s Plan: 
{leader_output}

User Profile Data:
{profile}

Task: 
{task}   

Answer:
'''


worker_search_leader_prompt='''You are a capable and resourceful subordinate who needs guidance from a knowledgeable manager to successfully complete an important task. You possess all relevant user data related to the task, while your manager does not have access to this information. Your objective is to select the leadership style your manager should adopt to provide you with the best plan for task completion. There are four specific leadership styles to choose from: [\"Directive\", \"Supportive\", \"Participative\", \"Achievement-Oriented\"]. Carefully consider which style is most appropriate and select only one. It is essential that your response consists solely of the chosen leadership style without any explanation, justification, or additional content. Your answer must be concise and strictly limited to one of the four options provided.

User Profile Data:
{profile}

Task: 
{task} 

Answer:
'''

def model_generate(model,tokenizer,format_prompt,max_tokens=512,ppl=False,n=1,lora_path = ""):
    if n==1:
        if ppl == False:
            sampling_params = vllm.SamplingParams(
                                    n=n, 
                                    top_p=0.9, 
                                    temperature=0,
                                    seed=666, 
                                    max_tokens=max_tokens,
                                    skip_special_tokens=True,  
                                    # stop_token_ids=stop_token_ids
                                    )
        else:
            sampling_params = vllm.SamplingParams(
                                    n=n, 
                                    top_p=0.9, 
                                    temperature=0,
                                    seed=666, 
                                    max_tokens=max_tokens,
                                    skip_special_tokens=True,  
                                    logprobs=True,
                                    )
    else:
        sampling_params = vllm.SamplingParams(
                                    n=n, 
                                    top_p=0.9, 
                                    top_k = 2,
                                    temperature=0.5,
                                    seed=666, 
                                    max_tokens=max_tokens,
                                    repetition_penalty=1.0,  
                                    skip_special_tokens=True,  
                                    # stop_token_ids=stop_token_ids
                                    )
    try:
        user_input = tokenizer.apply_chat_template(
                                    format_prompt,
                                    tokenize=False,
                                    add_generation_prompt=True
                            )
    except:
        user_input = [item[1]['content'] for item in format_prompt]
    if lora_path!="":
        responses = model.generate(user_input,
                                    sampling_params,
                                    use_tqdm = False,
                                    lora_request=LoRARequest("fintuen_v1", 1,lora_path),
                                )
        print("load_lora",lora_path)
    else:
        responses = model.generate(user_input,
                                    sampling_params,
                                    use_tqdm = False,
                                )
    # print(responses)
    res = []
    ppl_res = []
    for response in responses:
        res_text = response.outputs[n-1].text
        if ppl==True:
            # log_probs = response.log_probs 
            # tokens = response.input_tokens 
            # log_likelihood = sum(log_probs[:-1])
            if len(response.outputs[n-1].token_ids)==0:
                print(len(response.outputs[n-1].token_ids))
                continue
            perplexity = response.outputs[n-1].cumulative_logprob/len(response.outputs[n-1].token_ids)
            ppl_res.append(perplexity)
        # res_text = response.outputs[0].text.strip().strip('\n').split('\n')[0]
        print(res_text)
        print('#'*20)
        res.append(res_text) 
    return res,ppl_res

def model_generate_BBT(model,tokenizer,format_prompt,trained_prompt,max_tokens=512):

        
    user_input = tokenizer.apply_chat_template(
                                format_prompt,
                                tokenize=False,
                                add_generation_prompt=True
                            )

    bbt_input = []
    input_ids = tokenizer.tokenize(user_input, return_tensors="pt")["input_ids"].to(model.device)
    # input_ids = tokenizer.convert_tokens_to_ids(tokens).to(model.device)

    soft_embedding= trained_prompt.expand(input_ids.size(0), -1, -1)
    input_embeddings = model.transformer.embeddings.word_embeddings(input_ids)
    # out = model(torch.tensor([input_ids]).cuda(),output_hidden_states=True)
    # inputs_embeds=out.hidden_states[0]
    model_inputs = torch.cat([soft_embedding, input_embeddings], dim=0)
    
    bbt_input.append(input_ids)
    responses = model.generate(
            bbt_input,
            max_new_tokens=max_tokens
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def model_generate_pure(model,tokenizer,format_prompt):
    
    sampling_params = vllm.SamplingParams(
                            n=1, 
                            top_p=0.9, 
                            temperature=0,
                            seed=666, 
                            max_tokens=512,
                            skip_special_tokens=True,  
                            # stop_token_ids=stop_token_ids
                            )
    user_input = format_prompt
    responses = model.generate(user_input,
                                sampling_params,
                                use_tqdm = False,
                            )
    res = []
    for response in responses:
        res_text = response.outputs[0].text
        # res_text = response.outputs[0].text.strip().strip('\n').split('\n')[0]
        print('#'*20)
        res.append(res_text) 
    return res

def format_data_with_server_LLM(data,data_type='write'):
    if data_type=='write':
        format_input = []
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = skeleton_extract_template_ctx.format(question=user_input)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    elif data_type=='rec':
        format_input = []
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = skeleton_extract_template_ctx_rec.format(question=user_input)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    return format_input


def format_data_with_device_SLM(data,data_type='write'):
    if data_type=='write':
        format_input = []
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = ctx_few_shot['w_context'].format(
                    profile=sample["additional_profile"],
                    history="\n\n".join([f"Task: {p['title']}\nContent: {str(p['text'])}" for p in sample['profile']]),
                    task=sample["server_model_output"]
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    elif data_type=='rec':
        format_input = []
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = ctx_few_shot_rec['w_context'].format(
                    profile=sample["additional_profile"],
                    task=sample["server_model_output"]
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    return format_input


def format_data_with_device_SLM_ori(data,data_type = 'write'):
    if data_type == 'write':
        format_input = []
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = ctx_few_shot['w_context'].format(
                    profile=sample["additional_profile"],
                    history="\n\n".join([f"Task: {p['title']}\nContent: {str(p['text'])}" for p in sample['profile']]),
                    task=user_input
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    elif data_type == 'rec':
        format_input = []
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = ctx_few_shot_rec['w_context'].format(
                    profile=sample["additional_profile"],
                    task=user_input
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    return format_input



def evaluate_model(model_output,ground_truth):
    '''
    model_output = list(str) ['a','b']
    ground_truth = list(list(str)) [['a'],['b']]
    '''
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load('rouge')
    bleu_metric_res = bleu_metric.compute(predictions=model_output, references=ground_truth)
    rouge_metric_res = rouge_metric.compute(predictions=model_output, references=ground_truth)
    bleu_score = bleu_metric_res['score']
    rouge_1 = rouge_metric_res['rouge1']
    rouge_2 = rouge_metric_res['rouge2']
    rouge_l = rouge_metric_res['rougeL']
    return bleu_score,rouge_1,rouge_2,rouge_l

def my_format_data_with_server_LLM(data):
    format_input = []
    for sample in data:
        user_input = sample["conversations"][0]["value"]
        user_input = leader_prompt.format(
                exam1 = exam_of_leader,
                exam2 = exam_of_leader_2,
                task=user_input
                )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
            ]
        format_input.append(messages)
    return format_input

def my_format_data_with_server_dpo(data):
    format_input = []
    for sample in data:
        user_input = sample["conversations"][0]["value"]
        user_input = leader_prompt_dpo.format(
                task=user_input
                )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
            ]
        format_input.append(messages)
    return format_input


def my_format_data_with_server_dpo_rec(data):
    format_input = []
    for sample in data:
        user_input = sample["conversations"][0]["value"]
        user_input = leader_prompt_dpo_rec.format(
                task=user_input
                )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
            ]
        format_input.append(messages)
    return format_input

def my_format_data_with_server_LLM_leader_style(data,leader_style='',is_lora = False,data_type='write'):
    if data_type=='write':
        format_input = []
        if leader_style!='':
            leader_prompt_current = leader_hard_class[leader_style]
            leader_prompt_current = leader_prompt_current + leader_ICL
        else:
            leader_prompt_current = leader_prompt
            # leader_prompt_current = leader_prompt_v0
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = leader_prompt_current.format(
                    exam1 = exam_of_leader,
                    exam2 = exam_of_leader_2,
                    task=user_input
                    )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    elif data_type=='rec':
        format_input = []
        if leader_style!='':
            leader_prompt_current = leader_hard_class_rec[leader_style]
            leader_prompt_current = leader_prompt_current + leader_ICL
        else:
            leader_prompt_current = leader_prompt_rec
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = leader_prompt_current.format(
                    exam1 = exam_of_leader_rec,
                    exam2 = exam_of_leader_2_rec,
                    task=user_input
                    )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    return format_input



def my_format_data_with_device_SLM(data,data_type='write'):
    if data_type=='write':
        format_input = []
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = worker_prompt.format(
                    leader_output=sample["server_model_output"],
                    task=sample["conversations"][0]["value"],
                    profile=sample["additional_profile"],
                    history="\n\n".join([f"Task: {p['title']}\nContent: {str(p['text'])}" for p in sample['profile']]),
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    elif data_type=='rec':
        format_input = []
        for sample in data:
            user_input = sample["conversations"][0]["value"]
            user_input = worker_prompt_rec.format(
                    leader_output=sample["server_model_output"],
                    task=sample["conversations"][0]["value"],
                    profile=sample["additional_profile"],
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    return format_input

def prompt_search_leader_style_with_device_SLM(data):
    format_input = []
    for sample in data:
        user_input = sample["conversations"][0]["value"]
        user_input = worker_search_leader_prompt.format(
                task=sample["conversations"][0]["value"],
                profile=sample["additional_profile"],
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
            ]
        format_input.append(messages)
    return format_input




def chossen_template_LLM(data):
    res = []
    for sample in data:
        template = sample['server_model_output']
        res.append(template)
    return res

def choose_leader_style(text):
    leader_stye = ["Directive_Leadership","Supportive_Leadership","Participative_Leadership","Achievement_Oriented_Leadership"]
    leader_text_ls = ["Participative","Directive","Supportive","Achievement-Oriented"]
    ls =''
    for item in leader_text_ls:
        if item in text:
            if item =="Participative":
                ls = "Participative_Leadership"
            elif item =="Directive":
                ls = "Directive_Leadership"
            elif item =="Supportive":
                ls = "Supportive_Leadership"
            elif item =="Achievement-Oriented":
                ls = "Achievement_Oriented_Leadership"
            break
    return ls

def my_format_data_with_server_LLM_leader_style_choose(data,data_type='write'):
    
    if data_type=='write':
        format_input = []

        for sample in data:
            leader_style = choose_leader_style(sample['leader_style'])
            print(leader_style)
            if leader_style!='':
                leader_prompt_current = leader_hard_class[leader_style]
                leader_prompt_current = leader_prompt_current + leader_ICL
                print(leader_prompt_current)
            else:
                # leader_prompt_current = leader_prompt
                leader_prompt_current = leader_prompt_v0
            user_input = sample["conversations"][0]["value"]
            user_input = leader_prompt_current.format(
                    exam1 = exam_of_leader,
                    exam2 = exam_of_leader_2,
                    task=user_input
                    )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    elif data_type=='rec':
        format_input = []

        for sample in data:
            leader_style = choose_leader_style(sample['leader_style'])
            print(leader_style)
            if leader_style!='':
                leader_prompt_current = leader_hard_class_rec[leader_style]
                leader_prompt_current = leader_prompt_current + leader_ICL
            else:
                # leader_prompt_current = leader_prompt
                leader_prompt_current = leader_prompt_rec
            user_input = sample["conversations"][0]["value"]
            user_input = leader_prompt_current.format(
                    exam1 = exam_of_leader_rec,
                    exam2 = exam_of_leader_2_rec,
                    task=user_input
                    )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    return format_input


def choose_leader_style_rag_privacy(rag_ls):
    leader_stye = ["Directive_Leadership","Supportive_Leadership","Participative_Leadership","Achievement_Oriented_Leadership"]
    idx = [item['leader_style'] for item in rag_ls]
    counter = Counter(idx)
    most_common_element, count = counter.most_common(1)[0]
    return leader_stye[most_common_element]




def choose_leader_style_train(style_idx):
    leader_stye = ["Directive_Leadership","Supportive_Leadership","Participative_Leadership","Achievement_Oriented_Leadership"]
    return leader_stye[style_idx]


def my_format_data_with_server_LLM_leader_style_choose_rag_privacy(data,data_type='write',RAG_num=5):
    if data_type=='write':
        format_input = []
        for sample in data:
            if 'rag_result' in sample.keys():
                leader_style = choose_leader_style_rag_privacy(sample['rag_result'][:RAG_num])
                sample['leader_style'] = leader_style
            elif "leader_style" in sample.keys():
                leader_style = choose_leader_style_train(sample['leader_style'])
                sample['leader_style'] = leader_style
            print(leader_style)
            if leader_style!='':
                leader_prompt_current = leader_hard_class[leader_style]
                # leader_prompt_current = leader_prompt_current + "\n\nTask: {task}"
                leader_prompt_current = leader_prompt_current + leader_ICL
            else:
                # leader_prompt_current = leader_prompt
                leader_prompt_current = leader_prompt_v0
            
            user_input = sample["conversations"][0]["value"]
            user_input = leader_prompt_current.format(
                    exam1 = exam_of_leader,
                    exam2 = exam_of_leader_2,
                    task=user_input
                    )
            print(user_input)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            sample['LLM_input'] = messages
            format_input.append(messages)
    elif data_type=='rec':
        format_input = []
        for sample in data:
            if 'rag_result' in sample.keys():
                leader_style = choose_leader_style_rag_privacy(sample['rag_result'][:RAG_num])
                sample['leader_style'] = leader_style
            elif "leader_style" in sample.keys():
                leader_style = choose_leader_style_train(sample['leader_style'])
                sample['leader_style'] = leader_style
            print(leader_style)
            if leader_style!='':
                leader_prompt_current = leader_hard_class_rec[leader_style]
                # leader_prompt_current = leader_prompt_current + "\n\nTask: {task}"
                leader_prompt_current = leader_prompt_current + leader_ICL
            else:
                # leader_prompt_current = leader_prompt
                leader_prompt_current = leader_prompt_rec
            user_input = sample["conversations"][0]["value"]
            user_input = leader_prompt_current.format(
                    exam1 = exam_of_leader_rec,
                    exam2 = exam_of_leader_2_rec,
                    task=user_input
                    )
            print(user_input)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
            sample['LLM_input'] = messages
    return data,format_input

def my_format_data_with_server_LLM_leader_style_choose_rag_privacy_dpo(data, data_type='write'):
    if data_type=='write':
        format_input = []
        for sample in data:
            if 'rag_result' in sample.keys():
                leader_style = choose_leader_style_rag_privacy(sample['rag_result'][:3])
            elif "leader_style" in sample.keys():
                leader_style = choose_leader_style_train(sample['leader_style'])
            print(leader_style)
            if leader_style!='':
                leader_prompt_current = leader_hard_class[leader_style]
                # leader_prompt_current = leader_prompt_current + "\n\nTask: {task}"
                leader_prompt_current = leader_prompt_current + "\n\nTask: {task}"
            else:
                # leader_prompt_current = leader_prompt
                leader_prompt_current = leader_prompt_v0
            user_input = sample["conversations"][0]["value"]
            user_input = leader_prompt_current.format(
                    exam1 = exam_of_leader,
                    exam2 = exam_of_leader_2,
                    task=user_input
                    )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    elif data_type=='rec':
        format_input = []
        for sample in data:
            if 'rag_result' in sample.keys():
                leader_style = choose_leader_style_rag_privacy(sample['rag_result'][:3])
            elif "leader_style" in sample.keys():
                leader_style = choose_leader_style_train(sample['leader_style'])
            print(leader_style)
            if leader_style!='':
                leader_prompt_current = leader_hard_class_rec[leader_style]
                # leader_prompt_current = leader_prompt_current + "\n\nTask: {task}"
                leader_prompt_current = leader_prompt_current + "\n\nTask: {task}"
            else:
                # leader_prompt_current = leader_prompt
                leader_prompt_current = leader_prompt_rec
            user_input = sample["conversations"][0]["value"]
            user_input = leader_prompt_current.format(
                    exam1 = exam_of_leader,
                    exam2 = exam_of_leader_2,
                    task=user_input
                    )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
                ]
            format_input.append(messages)
    return format_input

def my_format_data_with_server_leader(data):
    format_input = []
    for sample in data:
        leader_style = choose_leader_style_train(sample['leader_style'])
        leader_prompt_current = leader_hard_class[leader_style] + "\n\nTask: {task}"
        
        user_input = sample["conversations"][0]["value"]
        user_input = leader_prompt_current.format(
                task=user_input
                )
        print(user_input)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
            ]
        format_input.append(messages)
    return format_input




def my_format_data_with_distill_baseline(data):
    format_input = []
    for sample in data:
        user_input = sample["conversations"][0]["value"]
        user_input = ctx_few_shot['w_context'].format(
                profile=sample["additional_profile"],
                history="\n\n".join([f"Task: {p['title']}\nContent: {str(p['text'])}" for p in sample['profile']]),
                task=user_input
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
            ]
        format_input.append(messages)
    return format_input

def my_format_data_with_distill_baseline_rec(data):
    format_input = []
    for sample in data:
        user_input = sample["conversations"][0]["value"]
        user_input = ctx_few_shot_rec['w_context'].format(
                profile=sample["additional_profile"],
                task=user_input
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
            ]
        format_input.append(messages)
    return format_input