"""
A module for obtaining repo readme and language data from the github API.

Before using this module, read through it, and follow the instructions marked
TODO.

After doing so, run it like this:

    python acquire.py

To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
from bs4 import BeautifulSoup
from env import github_token, github_username
from random import randint
from time import sleep

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

def get_repo_names():
    '''
    This function creates a list of repository names. It iterates over the 
    first 20 pages for english repositories using Ruby, JavaScript, Python,
    Java, and C++, then returns the final list.
    '''
    # create an empty list
    names = []
    # set which languages to conduct the search over
    langs = ['Ruby', 'JavaScript', 'Python', 'Java', 'C++']
    # for each languagein the above list...
    for lang in langs:
        # set the url for the current language
        url = f'https://github.com/search?l={lang}&q=stars%3A%3E0&s=stars&type=Repositories?spoken_language_code=en'
        # make soup out of the page at that url
        soup = BeautifulSoup(requests.get(url).content, 'html.parser')
        # select the block of html with all the repository information in it
        repos = soup.select('a.v-align-middle')
        # for each repository that was just grabbed...
        for r in repos:
            # grab the name of it
            repo_name = r['href']
            # add it to the names list for the final output
            names.append(repo_name)
        # set the page number to use as a starting point
        page = 2
        # while we are not yet past page 20...
        while page <= 20:
            # set the url to specify the language and page we are on
            url = f'https://github.com/search?l={lang}&p={page}&q=stars%3A%3E0&s=stars&type=Repositories?spoken_language_code=en'
            # create soup out of that page
            soup = BeautifulSoup(requests.get(url).content, 'html.parser')
            # select for the repository information 
            repos = soup.select('a.v-align-middle')
            # for each repository in the list of repository information...
            for r in repos:
                # save the name of the repository
                repo_name = r['href']
                # add it to the names list for the final output
                names.append(repo_name)
            # print what page was just gathered and the total count of repositories
            # saved in the names list
            print(f'finishing page: {page} of {lang}. Gathered {len(names)} repos.')
            # add one to page to move to the next page
            page += 1
            # sleep for 5 seconds to not be throttled for scraping Github
            sleep(5)
            # sleep for 30 seconds every 5th page
            if len(names)%50 == 0:
                sleep(30)
    # return the final list of repository names
    return names

# call the get_repos_names function to the variable REPOS for future use
REPOS = get_repo_names()

# set the headers for identificaiton
headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
# print to show we gathered all the repo names
print(headers)
# if headers doesn't have the needed information...
if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    # stop the code and tell the user to add it
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    '''
    This function returns the page data for the passed url after checking for the 
    page status code.
    '''
    # save the page response
    response = requests.get(url, headers=headers)
    # get the page data in json format
    response_data = response.json()
    # if the page throws a load error...
    if response.status_code != 200:
        # sleep for 30 seconds
        sleep(30)
        # ask for the same information again
        response = requests.get(url, headers=headers)
        response_data = response.json()
        # check if the page is still throwing that error
        if response.status_code != 200:
            # stop the code and tell the user if there is still and error. Do not save the data.
            raise Exception(f"Error response from github api! status code: {response.status_code}, "
                            f"response: {json.dumps(response_data)}"
            )
    # return the page's data in json format
    return response_data


def get_repo_language(repo: str) -> str:
    '''
    This function grabs the language for the repository passed to it.
    '''
    # create the url to access the repo
    url = f"https://api.github.com/repos{repo}"
    # grab the page data that will contain the language
    repo_info = github_api_request(url)
    # if the info is in a dictionary...
    if type(repo_info) is dict:
        # cast it
        repo_info = cast(Dict, repo_info)
        # if language doesn't exist as a key...
        if "language" not in repo_info:
            # throw an error
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        # otherwise return the language
        return repo_info["language"]
    # if not a dictionary, throw an error
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    '''
    This function grabs the contents of the repository readme and returns it.
    '''
    # create the url to access the repo's contents
    url = f"https://api.github.com/repos{repo}/contents/"
    # save the contents
    contents = github_api_request(url)
    # if the contents are in the form of a list
    if type(contents) is list:
        # cast it
        contents = cast(List, contents)
        # return it
        return contents
    # if not, throw an error
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    # for each file in the list of files
    for file in files:
        # if the name of the file starts with readme
        if file["name"].lower().startswith("readme"):
            # return the file url
            return file["download_url"]
    # else return an empty string
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    # save the contents of the repo
    contents = get_repo_contents(repo)
    # save the url for the readme of that repo
    readme_download_url = get_readme_download_url(contents)
    # if the url is an empty string
    if readme_download_url == "":
        # then save the contents as an empty string too
        readme_contents = ""
    # if the url is NOT an empty string
    else:
        # save the contents
        readme_contents = requests.get(readme_download_url).text
    # return a dictionary with the repo name, language, and contents
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    # return the dictionary for every processed repo in the REPOS list
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    '''
    This function saves the scraped data into a json file.
    '''
    # save the dictionary
    data = scrape_github_data()
    # save it locally in a json file names data.json
    json.dump(data, open("data.json", "w"), indent=1)
