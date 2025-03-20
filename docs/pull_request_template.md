# **[Ticket number `JIRA`] Title:**
Eg. `[OC-00]` Ticket name matching with `JIRA`

Note: Use the same schema for the title of PR

# **Summary:**
Briefly describe here what was done, follow the OKR policy

For multiple tickets being tackled under a single PR
1. [OC-001] Ticket title 1
2. [OC-001] Ticket title 2
3. [OC-003] Ticket title 3

# **Setup Instructions:**
Please mention the steps required to verify the changes or run the code / feature

# **Author checklist:**
- [ ] I have added relevant reviewers, labels, summary, titel
- [ ] I have locally compiled and run the code for functional verification
- [ ] I have added useful comments in the code, docstrings for functions and classes
- [ ] I have passed the required inputs / values and paths as parameters and created relevant schemas
- [ ] I have not used any hard coded values or paths in the code/script
- [ ] I have updated the relevant READMEs, comments and API documentation
- [ ] I have updated the requirements.txt
- [ ] I have performed code linting. Pylint score is ____
- [ ] I have performed code formatting using python black (line length = 100)
- [ ] I have verified the functionality in isolated docker environment
- [ ] ~~I have crossed out irrelevant checklist items~~

**Note:** to check and item, put x in square brackets, similarly, remove x to uncheck

### WARNING: 
For the reviewer, be careful when the PR contains only a single commit
Double check the squash merge title and the final commit message, make sure it contains the `JIRA` ticket number

