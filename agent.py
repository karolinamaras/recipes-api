"""
A module that provides tools and agents for interacting with GitHub repositories,
such as retrieving pull request details, commit information, and reviewing pull
requests using AI agents.

The module relies on OpenAI for natural language processing and GitHub's API for
interacting with GitHub repositories. It implements a workflow with three agents:
ContextAgent, CommentorAgent, and ReviewAndPostingAgent, to facilitate contextual
understanding, commentary generation, and posting of reviews for pull requests.

Classes:
    - None

Functions:
    - get_pr_details: Retrieves detailed information about a specific pull request.
    - get_file_contents: Fetches the contents of a file from the repository.
    - get_commit_details: Provides details about a specific commit.
    - post_review_comment_to_pr: Posts a review comment to a pull request on GitHub.
    - add_context_to_state: Adds gathered context information to a context state.
    - add_comment_to_state: Saves a draft review comment into a context state.
    - add_review_to_state: Saves a final review comment into a context state.
"""
import asyncio
import os
from typing import Any, Optional, Union

import dotenv
from github import Auth
from github import Github
from github.PullRequestReview import PullRequestReview
from llama_index.core.agent import FunctionAgent, AgentWorkflow
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv()
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

# repo_url = "https://github.com/karolinamaras/recipes-api.git"
repo_url = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")
try:
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable is not set")
    auth = Auth.Token(github_token)
    github_client = Github(auth=auth)
except ValueError as e:
    print(f"Error: {e}")
    github_client = None
except Exception as e:
    print(f"Error authenticating with GitHub: {e}")
    github_client = None

repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"

if github_client is not None:
    repo = github_client.get_repo(full_repo_name)


async def get_pr_details(pr_numb: int):
    """Use this tool to get details about a specific PR in the repository"""
    if github_client is not None:
        pr = repo.get_pull(pr_numb)
        commit_shas = []
        commits = pr.get_commits()

        for c in commits:
            commit_shas.append(c.sha)
        return {
            "author": pr.user.login,
            "title": pr.title,
            "body": pr.body,
            "diff_url": pr.diff_url,
            "state": pr.state,
            "number": pr.number,
            "commit_SHAs": commit_shas
        }
    return None


def get_file_contents(file_path):
    """Use this tool to get the contents of a file in the repository"""
    if github_client is not None:
        try:
            content = repo.get_contents(file_path)
            return content.decoded_content.decode('utf-8')
        except Exception as e:
            return f"Error fetching file contents: {e}"
    return None


def get_commit_details(sha: str) -> Optional[Union[list[dict[str, Any]], str]]:
    """Use this tool to this tool to get the commit details """
    if github_client is not None:
        try:
            commit = repo.get_commit(sha)
            changed_files: list[dict[str, Any]] = []
            for f in commit.files:
                changed_files.append({
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "changes": f.changes,
                    "patch": f.patch,
                })
            return changed_files
        except Exception as e:
            return f"Error fetching commit details: {e}"
    return None

def post_review_comment_to_pr(pr_numb: int, final_comment: str) -> PullRequestReview:
    """Post the final review comment to the PR on GitHub"""
    pr = repo.get_pull(pr_numb)
    if pr is None:
        raise ValueError(f"PR {pr_numb} not found")
    try:
        return pr.create_review(body=final_comment,event="COMMENT")
    except Exception as e:
        raise Exception(f"Error posting review comment to PR {pr_numb}: {e}")


async def add_context_to_state(ctx: Context, name:str, gathered_contexts: dict[str, Any]) -> None:
    """Save the context gathered by the ContextAgent in the context state.
    This is the metadata about the repo and pull request, such as the PR details, commit details and changed files."""
    async with ctx.store.edit_state() as current_state:
        current_state['gathered_contexts'][name] = gathered_contexts

async def add_comment_to_state(ctx: Context, draft_comment: str) -> None:
    """Save the draft comment in the context state"""
    async with ctx.store.edit_state() as current_state:
        current_state["draft_comment"] = draft_comment

async def add_review_to_state(ctx: Context, final_review: str) -> None:
    """Add the final_review comment to the context state"""
    async with ctx.store.edit_state() as current_state:
        current_state["final_review"] = final_review


pr_details_tool = FunctionTool.from_defaults(
    get_pr_details,
)
file_content_tool = FunctionTool.from_defaults(
    get_file_contents,
)
commit_details_tool = FunctionTool.from_defaults(
    get_commit_details,
)
post_review_tool = FunctionTool.from_defaults(
    post_review_comment_to_pr,
)

context_gathering_prompt = """
You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
"""

context_agent = FunctionAgent(
    tools=[pr_details_tool, file_content_tool, commit_details_tool, add_context_to_state],
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context about the pull request and its changes, and hands off to the Commentor Agent for review commentary.",
    system_prompt=context_gathering_prompt,
    can_handoff_to = ["CommentorAgent"]
)

commentary_system_prompt: str = """
You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the Context Agent. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
 - You must hand off to the Review and Posting agent once you are done drafting a review.
"""
commentor_agent = FunctionAgent(
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment. Then hand off to the Review and Posting agent once done.",
    system_prompt=commentary_system_prompt,
    llm=llm,
    tools=[add_comment_to_state],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

reviewer_post_prompt = """
You are the Review and Posting agent. You must use the Commentor Agent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.  
 """
review_and_poster_agent = FunctionAgent(
    name="ReviewAndPostingAgent",
    description="Gets review from the CommentorAgent, reviews it and posts the final review comment to the PR on GitHub.",
    system_prompt=reviewer_post_prompt,
    llm=llm,
    tools=[add_review_to_state, post_review_tool],
    can_handoff_to = ["CommentorAgent"]
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_poster_agent],
    root_agent=review_and_poster_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)

async def main():
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    github_client.close()