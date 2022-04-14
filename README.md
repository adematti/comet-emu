# Give a Welcome to the *COMET*

| | |
| ---      | ---      |
| **Author**:  |  Alex E. et al. |
| **Source**  |  [Source code at GitLab](https://gitlab.com/aegge/pt-emulator)  |
| **Documentation**: | [Documentation at Readthedocs](https://gitlab.com/aegge/pt-emulator)  |
| **Installation**:  |  `pip install comet`       |

:dizzy: **COMET** - Cosmological Observables Modelled with/by/from Emulated Theory.
<dl>
  <span class="sd">    The emulator makes use of evolution mapping (`Sanchez 2020</span>
  <span class="sd">    &lt;https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123511&gt;`_,</span>
  <span class="sd">    `Sanchez et al 2021 &lt;https://arxiv.org/abs/2108.12710&gt;`_,) to compress</span>
  <span class="sd">    the information of evolution parameters :math:`\mathbf{\Theta_{e}}`</span>
  <span class="sd">    (e.g. :math:`h,\,\Omega_\mathrm{K},\,w_0,\,w_\mathrm{a},\,A_\mathrm{s},\,</span>
  <span class="sd">    \ldots`) into the single quantity :math:`\sigma_{12}`, defined as the rms</span>
  <span class="sd">    fluctuation of the linear density contrast :math:`\delta` within spheres</span>
  <span class="sd">    of radius :math:`R=8\,\mathrm{Mpc}`.</span>

  <span class="sd">    This parameter, together with the parameters affecting the shape of the</span>
  <span class="sd">    power spectrum :math:`\mathbf{\Theta_{s}}` (e.g.</span>
  <span class="sd">    :math:`\omega_\mathrm{b},\,\omega_\mathrm{c},\,n_\mathrm{s}`), and</span>
  <span class="sd">    the linear growth rate :math:`f`, are used as base of the emulator.</span>

  <span class="sd">    The redshift-dependency of the multipoles can also be treated similarly to</span>
  <span class="sd">    the impact that different evolution parameters have on the power spectrum,</span>
  <span class="sd">    that is, by a simple rescaling of the amplitude of the power spectrum in</span>
  <span class="sd">    order to match the desired value of :math:`\sigma_{12}`.</span>

  <span class="sd">    Internally to the emulator, the pair :math:`\left[k,P(k)\right]` is</span>
  <span class="sd">    expressed in :math:`\left[\mathrm{Mpc}^{-1},\mathrm{Mpc}^3\right]` units,</span>
  <span class="sd">    since this is the only set of units for which the evolution parameter</span>
  <span class="sd">    degeneracy is present. If the user wishes to use the more conventional</span>
  <span class="sd">    unit set :math:`\left[h\,\mathrm{Mpc}^{-1},h^{-3}\,\mathrm{Mpc}^3\right]`,</span>
  <span class="sd">    they can do so by specifying it in the proper class attribute flag. In this</span>
  <span class="sd">    case, the input/output are converted into :math:`\mathrm{Mpc}` units</span>
  <span class="sd">    before being used/returned.</span>

  <span class="sd">    Geometrical distortions (AP corrections) are included a posteriori without</span>
  <span class="sd">    the need of including them in the emulation. This process is carried out</span>
  <span class="sd">    by first reconstructing the full anisotropic 2d galaxy power spectrum</span>
  <span class="sd">    :math:`P_\mathrm{gg}(k,\mu)`, summing up all the even multipoles up to</span>
  <span class="sd">    :math:`\ell=6`, applying distortions to :math:`k` and :math:`\mu`, and then</span>
  <span class="sd">    projecting again over the Legendre polynomials.</span>
  <span class="sd">    &quot;&quot;&quot;</span>

</dl>
## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/aegge/pt-emulator.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/user/project/integrations/)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Automatically merge when pipeline succeeds](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://docs.gitlab.com/ee/user/clusters/agent/)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!).  Thank you to [makeareadme.com](https://gitlab.com/-/experiment/new_project_readme_content:fa992421708f1099f9f6ba1f18c7e9b2?https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
