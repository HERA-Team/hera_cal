

<!DOCTYPE html >
<html lang = "en" >
  <head >
    <meta charset = "utf-8" >

  <link crossorigin = "anonymous" href = "https://assets-cdn.github.com/assets/frameworks-148da7a2b8b9ad739a5a0b8b68683fed4ac7e50ce8185f17d86aa05e75ed376e.css" integrity = "sha256-FI2nori5rXOaWguLaGg/7UrH5QzoGF8X2GqgXnXtN24=" media = "all" rel = "stylesheet" / >
  <link crossorigin = "anonymous" href = "https://assets-cdn.github.com/assets/github-2975123dd81713ee1a968c1654ab2b03483e6c6c4199314fab632b0b39d1a0ce.css" integrity = "sha256-KXUSPdgXE+4alowWVKsrA0g+bGxBmTFPq2MrCznRoM4=" media = "all" rel = "stylesheet" / >

  <meta name = "viewport" content = "width=device-width" >

  <title > capo / hsa7458_v000.py at master · zakiali / capo < /title >
  <link rel = "search" type = "application/opensearchdescription+xml" href = "/opensearch.xml" title = "GitHub" >
  <link rel = "fluid-icon" href = "https://github.com/fluidicon.png" title = "GitHub" >
  <meta property = "fb:app_id" content = "1401488693436528" >

    <meta content = "https://avatars2.githubusercontent.com/u/940272?v=3&amp;s=400" property = "og:image" / > < meta content = "GitHub" property = "og:site_name" / > < meta content = "object" property = "og:type" / > < meta content = "zakiali/capo" property = "og:title" / > < meta content = "https://github.com/zakiali/capo" property = "og:url" / > < meta content = "capo - Calibration and Analysis of PAPER Observations" property = "og:description" / >

  <link rel = "assets" href = "https://assets-cdn.github.com/" >
  <link rel = "web-socket" href = "wss://live.github.com/_sockets/VjI6MTM4ODU2OTM2OjY1MzVkYjE0ZDZmOTA1NzQ2MDVhODc4Njc2ZTU5NzYxNjViZDM4Y2QzZjBjYzM0M2ViMTI1MTA0NWJmNTNjM2M=--8c88abe55748225796b08febf9c5bb771bd2d09f" >
  <meta name = "pjax-timeout" content = "1000" >
  <link rel = "sudo-modal" href = "/sessions/sudo_modal" >
  <meta name = "request-id" content = "E6FD:2F29:5CC7C0F:88E1867:592F3EBA" data - pjax - transient >

  <meta name = "selected-link" value = "repo_source" data - pjax - transient >

  <meta name = "google-site-verification" content = "KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU" >
<meta name = "google-site-verification" content = "ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA" >
    <meta name = "google-analytics" content = "UA-3769691-2" >

<meta content = "collector.githubapp.com" name = "octolytics-host" / > < meta content = "github" name = "octolytics-app-id" / > < meta content = "https://collector.githubapp.com/github-external/browser_event" name = "octolytics-event-url" / > < meta content = "E6FD:2F29:5CC7C0F:88E1867:592F3EBA" name = "octolytics-dimension-request_id" / > < meta content = "iad" name = "octolytics-dimension-region_edge" / > < meta content = "iad" name = "octolytics-dimension-region_render" / > < meta content = "9141156" name = "octolytics-actor-id" / > < meta content = "plaplant" name = "octolytics-actor-login" / > < meta content = "2256a1012cc458a6a67b3b18d67aa1b78c0a7c2d4e25f2a639ff40cec7ca4ce7" name = "octolytics-actor-hash" / >
<meta content = "/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data - pjax - transient = "true" name = "analytics-location" / >

  <meta class = "js-ga-set" name = "dimension1" content = "Logged In" >

      <meta name = "hostname" content = "github.com" >
  <meta name = "user-login" content = "plaplant" >

      <meta name = "expected-hostname" content = "github.com" >
    <meta name="js-proxy-site-detection-payload" content="NjA4YmFiZGFkMmNmMzhlMGFkYTIyNTFmNjU0MjNmYjk4ZGJhNjdkMGI1ZDM1Zjc4NDdmNjU0NWJiYWJiZjhlZXx7InJlbW90ZV9hZGRyZXNzIjoiOTYuMjQ1LjcxLjEwOSIsInJlcXVlc3RfaWQiOiJFNkZEOjJGMjk6NUNDN0MwRjo4OEUxODY3OjU5MkYzRUJBIiwidGltZXN0YW1wIjoxNDk2MjY4NDgzLCJob3N0IjoiZ2l0aHViLmNvbSJ9">


  <meta name="html-safe-nonce" content="6ee434f79f83e50b87648add44fdcd3c122edbc0">

  <meta http-equiv="x-pjax-version" content="5ea1e0e2fa353d3240cbcf0583d6d894">
  

    
  <meta name="description" content="capo - Calibration and Analysis of PAPER Observations">
  <meta name="go-import" content="github.com/zakiali/capo git https://github.com/zakiali/capo.git">

  <meta content="940272" name="octolytics-dimension-user_id" /><meta content="zakiali" name="octolytics-dimension-user_login" /><meta content="10748958" name="octolytics-dimension-repository_id" /><meta content="zakiali/capo" name="octolytics-dimension-repository_nwo" /><meta content="true" name="octolytics-dimension-repository_public" /><meta content="true" name="octolytics-dimension-repository_is_fork" /><meta content="3567151" name="octolytics-dimension-repository_parent_id" /><meta content="AaronParsons/capo" name="octolytics-dimension-repository_parent_nwo" /><meta content="151115" name="octolytics-dimension-repository_network_root_id" /><meta content="dannyjacobs/capo" name="octolytics-dimension-repository_network_root_nwo" /><meta content="false" name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" />
  <link href="https://github.com/zakiali/capo/commits/master.atom" rel="alternate" title="Recent Commits to capo:master" type="application/atom+xml">


    <link rel="canonical" href="https://github.com/zakiali/capo/blob/master/cals/hsa7458_v000.py" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://assets-cdn.github.com/pinned-octocat.svg" color="#000000">
  <link rel="icon" type="image/x-icon" href="https://assets-cdn.github.com/favicon.ico">

<meta name="theme-color" content="#1e2327">


  <meta name="u2f-support" content="true">

  </head>

  <body class="logged-in env-production emoji-size-boost page-blob">
    



  <div class="position-relative js-header-wrapper ">
    <a href="#start-of-content" tabindex="1" class="accessibility-aid js-skip-to-content">Skip to content</a>
    <div id="js-pjax-loader-bar" class="pjax-loader-bar"><div class="progress"></div></div>

    
    
    



        
<div class="header" role="banner">
  <div class="container clearfix">
    <a class="header-logo-invertocat" href="https://github.com/orgs/HERA-Team/dashboard" data-hotkey="g d" aria-label="Homepage" data-ga-click="Header, go to dashboard, icon:logo">
  <svg aria-hidden="true" class="octicon octicon-mark-github" height="32" version="1.1" viewBox="0 0 16 16" width="32"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>


        <div class="header-search scoped-search site-scoped-search js-site-search" role="search">
  <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/zakiali/capo/search" class="js-site-search-form" data-scoped-search-url="/zakiali/capo/search" data-unscoped-search-url="/search" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
    <label class="form-control header-search-wrapper js-chromeless-input-container">
        <a href="/zakiali/capo/blob/master/cals/hsa7458_v000.py" class="header-search-scope no-underline">This repository</a>
      <input type="text"
        class="form-control header-search-input js-site-search-focus js-site-search-field is-clearable"
        data-hotkey="s"
        name="q"
        value=""
        placeholder="Search"
        aria-label="Search this repository"
        data-unscoped-placeholder="Search GitHub"
        data-scoped-placeholder="Search"
        autocapitalize="off">
        <input type="hidden" class="js-site-search-type-field" name="type" >
    </label>
</form></div>


      <ul class="header-nav float-left" role="navigation">
        <li class="header-nav-item">
          <a href="/pulls" aria-label="Pull requests you created" class="js-selected-navigation-item header-nav-link" data-ga-click="Header, click, Nav menu - item:pulls context:user" data-hotkey="g p" data-selected-links="/pulls /pulls/assigned /pulls/mentioned /pulls">
            Pull requests
</a>        </li>
        <li class="header-nav-item">
          <a href="/issues" aria-label="Issues you created" class="js-selected-navigation-item header-nav-link" data-ga-click="Header, click, Nav menu - item:issues context:user" data-hotkey="g i" data-selected-links="/issues /issues/assigned /issues/mentioned /issues">
            Issues
</a>        </li>
            <li class="header-nav-item">
              <a href="/marketplace" class="js-selected-navigation-item header-nav-link" data-ga-click="Header, click, Nav menu - item:marketplace context:user" data-selected-links=" /marketplace">
                Marketplace
</a>            </li>
          <li class="header-nav-item">
            <a class="header-nav-link" href="https://gist.github.com/" data-ga-click="Header, go to gist, text:gist">Gist</a>
          </li>
      </ul>

    
<ul class="header-nav user-nav float-right" id="user-links">
  <li class="header-nav-item">
    
    <a href="/notifications" aria-label="You have no unread notifications" class="header-nav-link notification-indicator tooltipped tooltipped-s js-socket-channel js-notification-indicator " data-channel="notification-changed:9141156" data-ga-click="Header, go to notifications, icon:read" data-hotkey="g n">
        <span class="mail-status "></span>
        <svg aria-hidden="true" class="octicon octicon-bell float-left" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M14 12v1H0v-1l.73-.58c.77-.77.81-2.55 1.19-4.42C2.69 3.23 6 2 6 2c0-.55.45-1 1-1s1 .45 1 1c0 0 3.39 1.23 4.16 5 .38 1.88.42 3.66 1.19 4.42l.66.58H14zm-7 4c1.11 0 2-.89 2-2H5c0 1.11.89 2 2 2z"/></svg>
</a>
  </li>

  <li class="header-nav-item dropdown js-menu-container">
    <a class="header-nav-link tooltipped tooltipped-s js-menu-target" href="/new"
       aria-label="Create new…"
       data-ga-click="Header, create new, icon:add">
      <svg aria-hidden="true" class="octicon octicon-plus float-left" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 9H7v5H5V9H0V7h5V2h2v5h5z"/></svg>
      <span class="dropdown-caret"></span>
    </a>

    <div class="dropdown-menu-content js-menu-content">
      <ul class="dropdown-menu dropdown-menu-sw">
        
<a class="dropdown-item" href="/new" data-ga-click="Header, create new repository">
  New repository
</a>

  <a class="dropdown-item" href="/new/import" data-ga-click="Header, import a repository">
    Import repository
  </a>

<a class="dropdown-item" href="https://gist.github.com/" data-ga-click="Header, create new gist">
  New gist
</a>

  <a class="dropdown-item" href="/organizations/new" data-ga-click="Header, create new organization">
    New organization
  </a>




      </ul>
    </div>
  </li>

  <li class="header-nav-item dropdown js-menu-container">
    <a class="header-nav-link name tooltipped tooltipped-sw js-menu-target" href="/plaplant"
       aria-label="View profile and more"
       data-ga-click="Header, show menu, icon:avatar">
      <img alt="@plaplant" class="avatar" src="https://avatars1.githubusercontent.com/u/9141156?v=3&amp;s=40" height="20" width="20">
      <span class="dropdown-caret"></span>
    </a>

    <div class="dropdown-menu-content js-menu-content">
      <div class="dropdown-menu dropdown-menu-sw">
        <div class="dropdown-header header-nav-current-user css-truncate">
          Signed in as <strong class="css-truncate-target">plaplant</strong>
        </div>

        <div class="dropdown-divider"></div>

        <a class="dropdown-item" href="/plaplant" data-ga-click="Header, go to profile, text:your profile">
          Your profile
        </a>
        <a class="dropdown-item" href="/plaplant?tab=stars" data-ga-click="Header, go to starred repos, text:your stars">
          Your stars
        </a>
        <a class="dropdown-item" href="/explore" data-ga-click="Header, go to explore, text:explore">
          Explore
        </a>
        <a class="dropdown-item" href="https://help.github.com" data-ga-click="Header, go to help, text:help">
          Help
        </a>

        <div class="dropdown-divider"></div>

        <a class="dropdown-item" href="/settings/profile" data-ga-click="Header, go to settings, icon:settings">
          Settings
        </a>

        <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/logout" class="logout-form" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="/FfMUI3WH+gIKYShwxSWHxKfE5YmA1X4SHwCX+wLHIW7CslJwyWWx2B4LTjajUEl1nEv/ZNXwZ/lecTYc9r6YQ==" /></div>
          <button type="submit" class="dropdown-item dropdown-signout" data-ga-click="Header, sign out, icon:logout">
            Sign out
          </button>
</form>      </div>
    </div>
  </li>
</ul>


    <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/logout" class="sr-only right-0" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="v5swIqOlKEWGMcSlQ4zD+ubY8Fiqp/VH/2/JiY8vx3z4xjU77Vahau5gbTxaFRTAIjbMMx/zYSBSag8OEP4hmA==" /></div>
      <button type="submit" class="dropdown-item dropdown-signout" data-ga-click="Header, sign out, icon:logout">
        Sign out
      </button>
</form>  </div>
</div>


      

  </div>

  <div id="start-of-content" class="accessibility-aid"></div>

    <div id="js-flash-container">
</div>



  <div role="main">
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode">
    <div id="js-repo-pjax-container" data-pjax-container>
        



    <div class="pagehead repohead instapaper_ignore readability-menu experiment-repo-nav">
      <div class="container repohead-details-container">

        <ul class="pagehead-actions">
  <li>
        <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/notifications/subscribe" class="js-social-container" data-autosubmit="true" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="1/CybUwTIvIBoZQDITHDL/VbQb2ZVzUKdwDXYcHwieDnxB7BqQbHcTB4mnP7+fFROvP7YQcNcYGUWwLzFTZXwg==" /></div>      <input class="form-control" id="repository_id" name="repository_id" type="hidden" value="10748958" />

        <div class="select-menu js-menu-container js-select-menu">
          <a href="/zakiali/capo/subscription"
            class="btn btn-sm btn-with-count select-menu-button js-menu-target" role="button" tabindex="0" aria-haspopup="true"
            data-ga-click="Repository, click Watch settings, action:blob#show">
            <span class="js-select-button">
                <svg aria-hidden="true" class="octicon octicon-eye" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                Watch
            </span>
          </a>
            <a class="social-count js-social-count"
              href="/zakiali/capo/watchers"
              aria-label="1 user is watching this repository">
              1
            </a>

        <div class="select-menu-modal-holder">
          <div class="select-menu-modal subscription-menu-modal js-menu-content">
            <div class="select-menu-header js-navigation-enable" tabindex="-1">
              <svg aria-label="Close" class="octicon octicon-x js-menu-close" height="16" role="img" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
              <span class="select-menu-title">Notifications</span>
            </div>

              <div class="select-menu-list js-navigation-container" role="menu">

                <div class="select-menu-item js-navigation-item selected" role="menuitem" tabindex="0">
                  <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
                  <div class="select-menu-item-text">
                    <input checked="checked" id="do_included" name="do" type="radio" value="included" />
                    <span class="select-menu-item-heading">Not watching</span>
                    <span class="description">Be notified when participating or @mentioned.</span>
                    <span class="js-select-button-text hidden-select-button-text">
                      <svg aria-hidden="true" class="octicon octicon-eye" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                      Watch
                    </span>
                  </div>
                </div>

                <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
                  <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
                  <div class="select-menu-item-text">
                    <input id="do_subscribed" name="do" type="radio" value="subscribed" />
                    <span class="select-menu-item-heading">Watching</span>
                    <span class="description">Be notified of all conversations.</span>
                    <span class="js-select-button-text hidden-select-button-text">
                      <svg aria-hidden="true" class="octicon octicon-eye" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                        Unwatch
                    </span>
                  </div>
                </div>

                <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
                  <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
                  <div class="select-menu-item-text">
                    <input id="do_ignore" name="do" type="radio" value="ignore" />
                    <span class="select-menu-item-heading">Ignoring</span>
                    <span class="description">Never be notified.</span>
                    <span class="js-select-button-text hidden-select-button-text">
                      <svg aria-hidden="true" class="octicon octicon-mute" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8 2.81v10.38c0 .67-.81 1-1.28.53L3 10H1c-.55 0-1-.45-1-1V7c0-.55.45-1 1-1h2l3.72-3.72C7.19 1.81 8 2.14 8 2.81zm7.53 3.22l-1.06-1.06-1.97 1.97-1.97-1.97-1.06 1.06L11.44 8 9.47 9.97l1.06 1.06 1.97-1.97 1.97 1.97 1.06-1.06L13.56 8l1.97-1.97z"/></svg>
                        Stop ignoring
                    </span>
                  </div>
                </div>

              </div>

            </div>
          </div>
        </div>
</form>
  </li>

  <li>
      <div class="js-toggler-container js-social-container starring-container ">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/zakiali/capo/unstar" class="starred" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="8ekoPSC6QLVmHNGedu7Dnu/z9wwF/SIlwdOCQQRiPC1HfaULU6pnGImaQkShttr56pFsT5unOOgx1cJrPC+nsw==" /></div>
      <button
        type="submit"
        class="btn btn-sm btn-with-count js-toggler-target"
        aria-label="Unstar this repository" title="Unstar zakiali/capo"
        data-ga-click="Repository, click unstar button, action:blob#show; text:Unstar">
        <svg aria-hidden="true" class="octicon octicon-star" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74z"/></svg>
        Unstar
      </button>
        <a class="social-count js-social-count" href="/zakiali/capo/stargazers"
           aria-label="0 users starred this repository">
          0
        </a>
</form>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/zakiali/capo/star" class="unstarred" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="WLmfScLe0+E1IHb/Nz/SRa3t5yU7b2uJwtfcpqKVxZy0/n61AjT10M6uPGXbQpXaB7nNm40JfZxfLxR5TTYBOg==" /></div>
      <button
        type="submit"
        class="btn btn-sm btn-with-count js-toggler-target"
        aria-label="Star this repository" title="Star zakiali/capo"
        data-ga-click="Repository, click star button, action:blob#show; text:Star">
        <svg aria-hidden="true" class="octicon octicon-star" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74z"/></svg>
        Star
      </button>
        <a class="social-count js-social-count" href="/zakiali/capo/stargazers"
           aria-label="0 users starred this repository">
          0
        </a>
</form>  </div>

  </li>

  <li>
          <a href="#fork-destination-box" class="btn btn-sm btn-with-count"
              title="Fork your own copy of zakiali/capo to your account"
              aria-label="Fork your own copy of zakiali/capo to your account"
              rel="facebox"
              data-ga-click="Repository, show fork modal, action:blob#show; text:Fork">
              <svg aria-hidden="true" class="octicon octicon-repo-forked" height="16" version="1.1" viewBox="0 0 10 16" width="10"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
            Fork
          </a>

          <div id="fork-destination-box" style="display: none;">
            <h2 class="facebox-header" data-facebox-id="facebox-header">Where should we fork this repository?</h2>
            <include-fragment src=""
                class="js-fork-select-fragment fork-select-fragment"
                data-url="/zakiali/capo/fork?fragment=1">
              <img alt="Loading" height="64" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-128.gif" width="64" />
            </include-fragment>
          </div>

    <a href="/zakiali/capo/network" class="social-count"
       aria-label="26 users forked this repository">
      26
    </a>
  </li>
</ul>

        <h1 class="public ">
  <svg aria-hidden="true" class="octicon octicon-repo-forked" height="16" version="1.1" viewBox="0 0 10 16" width="10"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
  <span class="author" itemprop="author"><a href="/zakiali" class="url fn" rel="author">zakiali</a></span><!--
--><span class="path-divider">/</span><!--
--><strong itemprop="name"><a href="/zakiali/capo" data-pjax="#js-repo-pjax-container">capo</a></strong>

    <span class="fork-flag">
      <span class="text">forked from <a href="/AaronParsons/capo">AaronParsons/capo</a></span>
    </span>
</h1>

      </div>
      <div class="container">
        
<nav class="reponav js-repo-nav js-sidenav-container-pjax"
     itemscope
     itemtype="http://schema.org/BreadcrumbList"
     role="navigation"
     data-pjax="#js-repo-pjax-container">

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a href="/zakiali/capo" class="js-selected-navigation-item selected reponav-item" data-hotkey="g c" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches /zakiali/capo" itemprop="url">
      <svg aria-hidden="true" class="octicon octicon-code" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M9.5 3L8 4.5 11.5 8 8 11.5 9.5 13 14 8 9.5 3zm-5 0L0 8l4.5 5L6 11.5 2.5 8 6 4.5 4.5 3z"/></svg>
      <span itemprop="name">Code</span>
      <meta itemprop="position" content="1">
</a>  </span>


  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a href="/zakiali/capo/pulls" class="js-selected-navigation-item reponav-item" data-hotkey="g p" data-selected-links="repo_pulls /zakiali/capo/pulls" itemprop="url">
      <svg aria-hidden="true" class="octicon octicon-git-pull-request" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M11 11.28V5c-.03-.78-.34-1.47-.94-2.06C9.46 2.35 8.78 2.03 8 2H7V0L4 3l3 3V4h1c.27.02.48.11.69.31.21.2.3.42.31.69v6.28A1.993 1.993 0 0 0 10 15a1.993 1.993 0 0 0 1-3.72zm-1 2.92c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zM4 3c0-1.11-.89-2-2-2a1.993 1.993 0 0 0-1 3.72v6.56A1.993 1.993 0 0 0 2 15a1.993 1.993 0 0 0 1-3.72V4.72c.59-.34 1-.98 1-1.72zm-.8 10c0 .66-.55 1.2-1.2 1.2-.65 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
      <span itemprop="name">Pull requests</span>
      <span class="Counter">0</span>
      <meta itemprop="position" content="3">
</a>  </span>

    <a href="/zakiali/capo/projects" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /zakiali/capo/projects">
      <svg aria-hidden="true" class="octicon octicon-project" height="16" version="1.1" viewBox="0 0 15 16" width="15"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z"/></svg>
      Projects
      <span class="Counter" >0</span>
</a>
    <a href="/zakiali/capo/wiki" class="js-selected-navigation-item reponav-item" data-hotkey="g w" data-selected-links="repo_wiki /zakiali/capo/wiki">
      <svg aria-hidden="true" class="octicon octicon-book" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M3 5h4v1H3V5zm0 3h4V7H3v1zm0 2h4V9H3v1zm11-5h-4v1h4V5zm0 2h-4v1h4V7zm0 2h-4v1h4V9zm2-6v9c0 .55-.45 1-1 1H9.5l-1 1-1-1H2c-.55 0-1-.45-1-1V3c0-.55.45-1 1-1h5.5l1 1 1-1H15c.55 0 1 .45 1 1zm-8 .5L7.5 3H2v9h6V3.5zm7-.5H9.5l-.5.5V12h6V3z"/></svg>
      Wiki
</a>

    <div class="reponav-dropdown js-menu-container">
      <button type="button" class="btn-link reponav-item reponav-dropdown js-menu-target " data-no-toggle aria-expanded="false" aria-haspopup="true">
        Insights
        <svg aria-hidden="true" class="octicon octicon-triangle-down v-align-middle text-gray" height="11" version="1.1" viewBox="0 0 12 16" width="8"><path fill-rule="evenodd" d="M0 5l6 6 6-6z"/></svg>
      </button>
      <div class="dropdown-menu-content js-menu-content">
        <div class="dropdown-menu dropdown-menu-sw">
          <a class="dropdown-item" href="/zakiali/capo/pulse" data-skip-pjax>
            <svg aria-hidden="true" class="octicon octicon-pulse" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M11.5 8L8.8 5.4 6.6 8.5 5.5 1.6 2.38 8H0v2h3.6l.9-1.8.9 5.4L9 8.5l1.6 1.5H14V8z"/></svg>
            Pulse
          </a>
          <a class="dropdown-item" href="/zakiali/capo/graphs" data-skip-pjax>
            <svg aria-hidden="true" class="octicon octicon-graph" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M16 14v1H0V0h1v14h15zM5 13H3V8h2v5zm4 0H7V3h2v10zm4 0h-2V6h2v7z"/></svg>
            Graphs
          </a>
        </div>
      </div>
    </div>
</nav>

      </div>
    </div>

<div class="container new-discussion-timeline experiment-repo-nav">
  <div class="repository-content">

    
          

<a href="/zakiali/capo/blob/50a65799717aba6bc99acc3a3029459c39bb0de3/cals/hsa7458_v000.py" class="d-none js-permalink-shortcut" data-hotkey="y">Permalink</a>

<!-- blob contrib key: blob_contributors:v21:2db68b5c5bc5282c06230b1d874a5ff5 -->

<div class="file-navigation js-zeroclipboard-container">
  
<div class="select-menu branch-select-menu js-menu-container js-select-menu float-left">
  <button class=" btn btn-sm select-menu-button js-menu-target css-truncate" data-hotkey="w"
    
    type="button" aria-label="Switch branches or tags" tabindex="0" aria-haspopup="true">
      <i>Branch:</i>
      <span class="js-select-button css-truncate-target">master</span>
  </button>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax>

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <svg aria-label="Close" class="octicon octicon-x js-menu-close" height="16" role="img" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
        <span class="select-menu-title">Switch branches/tags</span>
      </div>

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Filter branches/tags" id="context-commitish-filter-field" class="form-control js-filterable-field js-navigation-enable" placeholder="Filter branches/tags">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" data-filter-placeholder="Filter branches/tags" class="js-select-menu-tab" role="tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" data-filter-placeholder="Find a tag…" class="js-select-menu-tab" role="tab">Tags</a>
            </li>
          </ul>
        </div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches" role="menu">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/zakiali/capo/blob/CAPO_online/cals/hsa7458_v000.py"
               data-name="CAPO_online"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                CAPO_online
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open selected"
               href="/zakiali/capo/blob/master/cals/hsa7458_v000.py"
               data-name="master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                master
              </span>
            </a>
        </div>

          <div class="select-menu-no-results">Nothing to show</div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div>

    </div>
  </div>
</div>

  <div class="BtnGroup float-right">
    <a href="/zakiali/capo/find/master"
          class="js-pjax-capture-input btn btn-sm BtnGroup-item"
          data-pjax
          data-hotkey="t">
      Find file
    </a>
    <button aria-label="Copy file path to clipboard" class="js-zeroclipboard btn btn-sm BtnGroup-item tooltipped tooltipped-s" data-copied-hint="Copied!" type="button">Copy path</button>
  </div>
  <div class="breadcrumb js-zeroclipboard-target">
    <span class="repo-root js-repo-root"><span class="js-path-segment"><a href="/zakiali/capo"><span>capo</span></a></span></span><span class="separator">/</span><span class="js-path-segment"><a href="/zakiali/capo/tree/master/cals"><span>cals</span></a></span><span class="separator">/</span><strong class="final-path">hsa7458_v000.py</strong>
  </div>
</div>


<include-fragment class="commit-tease" src="/zakiali/capo/contributors/master/cals/hsa7458_v000.py">
  <div>
    Fetching contributors&hellip;
  </div>

  <div class="commit-tease-contributors">
    <img alt="" class="loader-loading float-left" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32-EAF2F5.gif" width="16" />
    <span class="loader-error">Cannot retrieve contributors at this time</span>
  </div>
</include-fragment>
<div class="file">
  <div class="file-header">
  <div class="file-actions">

    <div class="BtnGroup">
      <a href="/zakiali/capo/raw/master/cals/hsa7458_v000.py" class="btn btn-sm BtnGroup-item" id="raw-url">Raw</a>
        <a href="/zakiali/capo/blame/master/cals/hsa7458_v000.py" class="btn btn-sm js-update-url-with-hash BtnGroup-item" data-hotkey="b">Blame</a>
      <a href="/zakiali/capo/commits/master/cals/hsa7458_v000.py" class="btn btn-sm BtnGroup-item" rel="nofollow">History</a>
    </div>

        <a class="btn-octicon tooltipped tooltipped-nw"
           href="github-mac://openRepo/https://github.com/zakiali/capo?branch=master&amp;filepath=cals%2Fhsa7458_v000.py"
           aria-label="Open this file in GitHub Desktop"
           data-ga-click="Repository, open with desktop, type:mac">
            <svg aria-hidden="true" class="octicon octicon-device-desktop" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M15 2H1c-.55 0-1 .45-1 1v9c0 .55.45 1 1 1h5.34c-.25.61-.86 1.39-2.34 2h8c-1.48-.61-2.09-1.39-2.34-2H15c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm0 9H1V3h14v8z"/></svg>
        </a>

        <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/zakiali/capo/edit/master/cals/hsa7458_v000.py" class="inline-form js-update-url-with-hash" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="E20jqbpsHuIvjgjlbxtD+QiRz+VwsrzE9cKK+HmBxX+6rMckoaKTWg6yeCq/Rdg2/b7ByQN1vNNsUyIhSgEoqA==" /></div>
          <button class="btn-octicon tooltipped tooltipped-nw" type="submit"
            aria-label="Edit the file in your fork of this project" data-hotkey="e" data-disable-with>
            <svg aria-hidden="true" class="octicon octicon-pencil" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"/></svg>
          </button>
</form>        <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/zakiali/capo/delete/master/cals/hsa7458_v000.py" class="inline-form" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="M5MIEJ2tlcVbNByEWwOgTVcdpBJyjobtnbP+Bebed37NZvZHNTsxAPxmPV2R7F/LL+1hTPUwtkKCeeB8EdoVdA==" /></div>
          <button class="btn-octicon btn-octicon-danger tooltipped tooltipped-nw" type="submit"
            aria-label="Delete the file in your fork of this project" data-disable-with>
            <svg aria-hidden="true" class="octicon octicon-trashcan" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M11 2H9c0-.55-.45-1-1-1H5c-.55 0-1 .45-1 1H2c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1v9c0 .55.45 1 1 1h7c.55 0 1-.45 1-1V5c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm-1 12H3V5h1v8h1V5h1v8h1V5h1v8h1V5h1v9zm1-10H2V3h9v1z"/></svg>
          </button>
</form>  </div>

  <div class="file-info">
      <span class="file-mode" title="File mode">executable file</span>
      <span class="file-info-divider"></span>
      280 lines (268 sloc)
      <span class="file-info-divider"></span>
    57 KB
  </div>
</div>

  

  <div itemprop="text" class="blob-wrapper data type-python">
      <table class="highlight tab-size js-file-line-container" data-tab-size="8">
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> aipy <span class="pl-k">as</span> a, numpy <span class="pl-k">as</span> n</td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line">cm_p_m <span class="pl-k">=</span> <span class="pl-c1">100</span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">AntennaArray</span>(<span class="pl-e">a</span>.<span class="pl-e">pol</span>.<span class="pl-e">AntennaArray</span>):</td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-k">*</span><span class="pl-smi">args</span>, <span class="pl-k">**</span><span class="pl-smi">kwargs</span>):</td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line">        a.pol.AntennaArray.<span class="pl-c1">__init__</span>(<span class="pl-c1">self</span>, <span class="pl-k">*</span>args, <span class="pl-k">**</span>kwargs)</td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.antpos_ideal <span class="pl-k">=</span> kwargs.pop(<span class="pl-s"><span class="pl-pds">&#39;</span>antpos_ideal<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">update_gains</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line">        gains <span class="pl-k">=</span> <span class="pl-c1">self</span>.gain <span class="pl-k">*</span> <span class="pl-c1">self</span>.amp_coeffs</td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> i,gain <span class="pl-k">in</span> <span class="pl-c1">zip</span>(<span class="pl-c1">self</span>.ant_layout.flatten(), gains.flatten()):</td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>[i].set_params({<span class="pl-s"><span class="pl-pds">&#39;</span>amp_x<span class="pl-pds">&#39;</span></span>:gain})</td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>[i].set_params({<span class="pl-s"><span class="pl-pds">&#39;</span>amp_y<span class="pl-pds">&#39;</span></span>:gain})</td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">update_delays</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line">        ns,ew <span class="pl-k">=</span> n.indices(<span class="pl-c1">self</span>.ant_layout.shape)</td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line">        dlys <span class="pl-k">=</span> ns<span class="pl-k">*</span><span class="pl-c1">self</span>.tau_ns <span class="pl-k">+</span> ew<span class="pl-k">*</span><span class="pl-c1">self</span>.tau_ew <span class="pl-k">+</span> <span class="pl-c1">self</span>.dly_coeffs</td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> i,tau <span class="pl-k">in</span> <span class="pl-c1">zip</span>(<span class="pl-c1">self</span>.ant_layout.flatten(), dlys.flatten()):</td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>[i].set_params({<span class="pl-s"><span class="pl-pds">&#39;</span>dly_x<span class="pl-pds">&#39;</span></span>:tau})</td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>[i].set_params({<span class="pl-s"><span class="pl-pds">&#39;</span>dly_y<span class="pl-pds">&#39;</span></span>:tau <span class="pl-k">+</span> <span class="pl-c1">self</span>.dly_xx_to_yy.flatten()[i]})</td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">update</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>        self.update_gains()</span></td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>        self.update_delays()</span></td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line">        a.pol.AntennaArray.update(<span class="pl-c1">self</span>)</td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_params</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">ant_prms</span><span class="pl-k">=</span>{<span class="pl-s"><span class="pl-pds">&#39;</span>*<span class="pl-pds">&#39;</span></span>:<span class="pl-s"><span class="pl-pds">&#39;</span>*<span class="pl-pds">&#39;</span></span>}):</td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">try</span>: prms <span class="pl-k">=</span> a.pol.AntennaArray.get_params(<span class="pl-c1">self</span>, ant_prms)</td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">except</span>(<span class="pl-c1">IndexError</span>): <span class="pl-k">return</span> {}</td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> k <span class="pl-k">in</span> ant_prms:</td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> k <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>aa<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-k">not</span> prms.has_key(<span class="pl-s"><span class="pl-pds">&#39;</span>aa<span class="pl-pds">&#39;</span></span>): prms[<span class="pl-s"><span class="pl-pds">&#39;</span>aa<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">for</span> val <span class="pl-k">in</span> ant_prms[k]:</td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">if</span>   val <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>tau_ns<span class="pl-pds">&#39;</span></span>: prms[<span class="pl-s"><span class="pl-pds">&#39;</span>aa<span class="pl-pds">&#39;</span></span>][<span class="pl-s"><span class="pl-pds">&#39;</span>tau_ns<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> <span class="pl-c1">self</span>.tau_ns</td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">elif</span> val <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>tau_ew<span class="pl-pds">&#39;</span></span>: prms[<span class="pl-s"><span class="pl-pds">&#39;</span>aa<span class="pl-pds">&#39;</span></span>][<span class="pl-s"><span class="pl-pds">&#39;</span>tau_ew<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> <span class="pl-c1">self</span>.tau_ew</td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">elif</span> val <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>gain<span class="pl-pds">&#39;</span></span>: prms[<span class="pl-s"><span class="pl-pds">&#39;</span>aa<span class="pl-pds">&#39;</span></span>][<span class="pl-s"><span class="pl-pds">&#39;</span>gain<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> <span class="pl-c1">self</span>.gain</td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">try</span>: top_pos <span class="pl-k">=</span> n.dot(<span class="pl-c1">self</span>._eq2zen, <span class="pl-c1">self</span>[<span class="pl-c1">int</span>(k)].pos)</td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line">                <span class="pl-c"><span class="pl-c">#</span> <span class="pl-k">XXX</span> should multiply this by len_ns to match set_params.</span></td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">except</span>(<span class="pl-c1">ValueError</span>): <span class="pl-k">continue</span></td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> ant_prms[k] <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>*<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line">                    prms[k].update({<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:top_pos[<span class="pl-c1">0</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:top_pos[<span class="pl-c1">1</span>], <span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:top_pos[<span class="pl-c1">2</span>]})</td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">for</span> val <span class="pl-k">in</span> ant_prms[k]:</td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">                        <span class="pl-k">if</span>   val <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>: prms[k][<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> top_pos[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">                        <span class="pl-k">elif</span> val <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>: prms[k][<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> top_pos[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line">                        <span class="pl-k">elif</span> val <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>: prms[k][<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> top_pos[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> prms</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_params</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">prms</span>):</td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">        changed <span class="pl-k">=</span> a.pol.AntennaArray.set_params(<span class="pl-c1">self</span>, prms)</td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> i, ant <span class="pl-k">in</span> <span class="pl-c1">enumerate</span>(<span class="pl-c1">self</span>):</td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">            ant_changed <span class="pl-k">=</span> <span class="pl-c1">False</span></td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">            top_pos <span class="pl-k">=</span> n.dot(<span class="pl-c1">self</span>._eq2zen, ant.pos)</td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">try</span>:</td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line">                top_pos[<span class="pl-c1">0</span>] <span class="pl-k">=</span> prms[<span class="pl-c1">str</span>(i)][<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>]</td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line">                ant_changed <span class="pl-k">=</span> <span class="pl-c1">True</span></td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">except</span>(<span class="pl-c1">KeyError</span>): <span class="pl-k">pass</span></td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">try</span>:</td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">                top_pos[<span class="pl-c1">1</span>] <span class="pl-k">=</span> prms[<span class="pl-c1">str</span>(i)][<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>]</td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">                ant_changed <span class="pl-k">=</span> <span class="pl-c1">True</span></td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">except</span>(<span class="pl-c1">KeyError</span>): <span class="pl-k">pass</span></td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">try</span>:</td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line">                top_pos[<span class="pl-c1">2</span>] <span class="pl-k">=</span> prms[<span class="pl-c1">str</span>(i)][<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>]</td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line">                ant_changed <span class="pl-k">=</span> <span class="pl-c1">True</span></td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">except</span>(<span class="pl-c1">KeyError</span>): <span class="pl-k">pass</span></td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> ant_changed: ant.pos <span class="pl-k">=</span> n.dot(n.linalg.inv(<span class="pl-c1">self</span>._eq2zen), top_pos) <span class="pl-k">/</span> a.const.len_ns <span class="pl-k">*</span> cm_p_m</td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">            changed <span class="pl-k">|=</span> ant_changed</td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">try</span>: <span class="pl-c1">self</span>.tau_ns, changed <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>aa<span class="pl-pds">&#39;</span></span>][<span class="pl-s"><span class="pl-pds">&#39;</span>tau_ns<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">except</span>(<span class="pl-c1">KeyError</span>): <span class="pl-k">pass</span></td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">try</span>: <span class="pl-c1">self</span>.tau_ew, changed <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>aa<span class="pl-pds">&#39;</span></span>][<span class="pl-s"><span class="pl-pds">&#39;</span>tau_ew<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">except</span>(<span class="pl-c1">KeyError</span>): <span class="pl-k">pass</span></td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">try</span>: <span class="pl-c1">self</span>.gain, changed <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>aa<span class="pl-pds">&#39;</span></span>][<span class="pl-s"><span class="pl-pds">&#39;</span>gain<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">except</span>(<span class="pl-c1">KeyError</span>): <span class="pl-k">pass</span></td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> changed: <span class="pl-c1">self</span>.update()</td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> changed</td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line">prms <span class="pl-k">=</span> {</td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;</span>loc<span class="pl-pds">&#39;</span></span>: (<span class="pl-s"><span class="pl-pds">&#39;</span>-30:43:17.5<span class="pl-pds">&#39;</span></span>, <span class="pl-s"><span class="pl-pds">&#39;</span>21:25:41.9<span class="pl-pds">&#39;</span></span>), <span class="pl-c"><span class="pl-c">#</span> KAT, SA (GPS)</span></td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;</span>antpos_ideal<span class="pl-pds">&#39;</span></span>: {</td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">80</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">14.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">24.2871131</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a0</span></td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">104</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">24.2871131</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a1</span></td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">96</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">14.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">24.2871131</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a2</span></td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">64</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">21.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">12.12435565</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a3</span></td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">53</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">7.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">12.12435565</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a4</span></td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">31</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">7.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">12.12435565</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a5</span></td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">65</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">21.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">12.12435565</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a5</span></td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">88</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">28.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a5</span></td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">9</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">14.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a5</span></td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">20</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a5</span></td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">89</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">14.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a5</span></td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">43</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">28.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a5</span></td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">105</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">21.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">12.12435565</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a3</span></td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">22</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">7.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">12.12435565</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a4</span></td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">81</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">7.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">12.12435565</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a5</span></td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">10</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">21.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">12.12435565</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a5</span></td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">72</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">14.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">24.2871131</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a0</span></td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">112</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">24.2871131</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a1</span></td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">97</span>: {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">14.0</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">24.2871131</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0.0</span> }, <span class="pl-c"><span class="pl-c">#</span>a2</span></td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">0</span>   :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">1</span>   :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">2</span>   :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">3</span>   :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">4</span>   :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">5</span>   :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">6</span>   :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">7</span>   :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">8</span>   :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">11</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">12</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">13</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">14</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">15</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">16</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">17</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">18</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">19</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">21</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">23</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">24</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">25</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">26</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">27</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">28</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">29</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">30</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">32</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">33</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">34</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">35</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">36</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">37</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">38</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">39</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">40</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">41</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">42</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">44</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">45</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">46</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">47</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">48</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">49</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">50</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">51</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">52</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">54</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">55</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">56</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">57</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">58</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">59</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">60</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">61</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">62</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">63</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">66</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">67</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">68</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">69</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">70</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">71</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">73</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">74</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">75</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">76</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">77</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">78</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">79</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">82</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">83</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">84</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">85</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">86</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">87</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">90</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L178" class="blob-num js-line-number" data-line-number="178"></td>
        <td id="LC178" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">91</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L179" class="blob-num js-line-number" data-line-number="179"></td>
        <td id="LC179" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">92</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L180" class="blob-num js-line-number" data-line-number="180"></td>
        <td id="LC180" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">93</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L181" class="blob-num js-line-number" data-line-number="181"></td>
        <td id="LC181" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">94</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L182" class="blob-num js-line-number" data-line-number="182"></td>
        <td id="LC182" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">95</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L183" class="blob-num js-line-number" data-line-number="183"></td>
        <td id="LC183" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">98</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L184" class="blob-num js-line-number" data-line-number="184"></td>
        <td id="LC184" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">99</span>  :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L185" class="blob-num js-line-number" data-line-number="185"></td>
        <td id="LC185" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">100</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L186" class="blob-num js-line-number" data-line-number="186"></td>
        <td id="LC186" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">101</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L187" class="blob-num js-line-number" data-line-number="187"></td>
        <td id="LC187" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">102</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L188" class="blob-num js-line-number" data-line-number="188"></td>
        <td id="LC188" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">103</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L189" class="blob-num js-line-number" data-line-number="189"></td>
        <td id="LC189" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">106</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L190" class="blob-num js-line-number" data-line-number="190"></td>
        <td id="LC190" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">107</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L191" class="blob-num js-line-number" data-line-number="191"></td>
        <td id="LC191" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">108</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L192" class="blob-num js-line-number" data-line-number="192"></td>
        <td id="LC192" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">109</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L193" class="blob-num js-line-number" data-line-number="193"></td>
        <td id="LC193" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">110</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L194" class="blob-num js-line-number" data-line-number="194"></td>
        <td id="LC194" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">111</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L195" class="blob-num js-line-number" data-line-number="195"></td>
        <td id="LC195" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">113</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L196" class="blob-num js-line-number" data-line-number="196"></td>
        <td id="LC196" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">114</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L197" class="blob-num js-line-number" data-line-number="197"></td>
        <td id="LC197" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">115</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L198" class="blob-num js-line-number" data-line-number="198"></td>
        <td id="LC198" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">116</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L199" class="blob-num js-line-number" data-line-number="199"></td>
        <td id="LC199" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">117</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L200" class="blob-num js-line-number" data-line-number="200"></td>
        <td id="LC200" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">118</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L201" class="blob-num js-line-number" data-line-number="201"></td>
        <td id="LC201" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">119</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L202" class="blob-num js-line-number" data-line-number="202"></td>
        <td id="LC202" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">120</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L203" class="blob-num js-line-number" data-line-number="203"></td>
        <td id="LC203" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">121</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L204" class="blob-num js-line-number" data-line-number="204"></td>
        <td id="LC204" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">122</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L205" class="blob-num js-line-number" data-line-number="205"></td>
        <td id="LC205" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">123</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L206" class="blob-num js-line-number" data-line-number="206"></td>
        <td id="LC206" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">124</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L207" class="blob-num js-line-number" data-line-number="207"></td>
        <td id="LC207" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">125</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L208" class="blob-num js-line-number" data-line-number="208"></td>
        <td id="LC208" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">126</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L209" class="blob-num js-line-number" data-line-number="209"></td>
        <td id="LC209" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">127</span> :{<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-k">-</span><span class="pl-c1">1</span> }, </td>
      </tr>
      <tr>
        <td id="L210" class="blob-num js-line-number" data-line-number="210"></td>
        <td id="LC210" class="blob-code blob-code-inner js-file-line">    },</td>
      </tr>
      <tr>
        <td id="L211" class="blob-num js-line-number" data-line-number="211"></td>
        <td id="LC211" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;</span>amps<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">dict</span>(<span class="pl-c1">zip</span>(<span class="pl-c1">range</span>(<span class="pl-c1">128</span>),n.ones(<span class="pl-c1">128</span>))),</td>
      </tr>
      <tr>
        <td id="L212" class="blob-num js-line-number" data-line-number="212"></td>
        <td id="LC212" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;</span>amp_coeffs<span class="pl-pds">&#39;</span></span>: n.array([<span class="pl-c1">1</span>]<span class="pl-k">*</span><span class="pl-c1">128</span>),</td>
      </tr>
      <tr>
        <td id="L213" class="blob-num js-line-number" data-line-number="213"></td>
        <td id="LC213" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;</span>bp_r<span class="pl-pds">&#39;</span></span>: n.array([[<span class="pl-c1">1</span>.]]<span class="pl-k">*</span><span class="pl-c1">128</span>),</td>
      </tr>
      <tr>
        <td id="L214" class="blob-num js-line-number" data-line-number="214"></td>
        <td id="LC214" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;</span>bp_i<span class="pl-pds">&#39;</span></span>: n.array([[<span class="pl-c1">0</span>.,<span class="pl-c1">0</span>.,<span class="pl-c1">0</span>.]]<span class="pl-k">*</span><span class="pl-c1">128</span>),</td>
      </tr>
      <tr>
        <td id="L215" class="blob-num js-line-number" data-line-number="215"></td>
        <td id="LC215" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;</span>twist<span class="pl-pds">&#39;</span></span>: n.array([<span class="pl-c1">0</span>]<span class="pl-k">*</span><span class="pl-c1">128</span>),</td>
      </tr>
      <tr>
        <td id="L216" class="blob-num js-line-number" data-line-number="216"></td>
        <td id="LC216" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;</span>beam<span class="pl-pds">&#39;</span></span>: a.fit.BeamAlm,</td>
      </tr>
      <tr>
        <td id="L217" class="blob-num js-line-number" data-line-number="217"></td>
        <td id="LC217" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;</span>bm_prms<span class="pl-pds">&#39;</span></span>: {</td>
      </tr>
      <tr>
        <td id="L218" class="blob-num js-line-number" data-line-number="218"></td>
        <td id="LC218" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;</span>alm7<span class="pl-pds">&#39;</span></span>: n.array(</td>
      </tr>
      <tr>
        <td id="L219" class="blob-num js-line-number" data-line-number="219"></td>
        <td id="LC219" class="blob-code blob-code-inner js-file-line">[-1081324093.5604355, 0.0, 0.0, 0.0, -262155437.14744619, -40523.293875736716, -1144639049.4450433, 0.0, 0.0, 0.0, -1209061567.4856892, -62596.955398614278, 0.0, 0.0, 234574880.79412466, -59373.276068766572, 0.0, 0.0, 204426224.06441718, 55114.563591448132, 1729080775.5577424, 0.0, 0.0, 0.0, -5627870.2693141159, 8441.4522962633619, 0.0, 0.0, -957871468.02796948, -26154.972628033676, 0.0, 0.0, -87102252.471384808, -50052.73968168487, 0.0, 0.0, 191828502.26962891, -10048.43132508696, 0.0, 0.0, 110776547.27628954, -37351.826990050504, -906797537.4834379, 0.0, 0.0, 0.0, 329222817.13445771, -29060.739352269207, 0.0, 0.0, 546809652.71201193, -115946.69818003669, 0.0, 0.0, -591134355.63541889, 62683.080456769902, 0.0, 0.0, -144055745.14333308, -3344.9767162177391, 0.0, 0.0, -368656232.30881768, -1608.1983129602879, 0.0, 0.0, -118801805.27812727, 31905.336279017167, 0.0, 0.0, -70225720.747015119, -37865.877995669623, 235419758.4512482, 0.0, 0.0, 0.0, 361579085.9617995, 79072.147640181458, 0.0, 0.0, 146429964.96514896, -102117.14213434084, 0.0, 0.0, 106427393.26674381, -60107.204182634705, 0.0, 0.0, 59154166.248101547, -20388.350978482271, 0.0, 0.0, 176029459.26438314, 33285.591239419438, 0.0, 0.0, 527618634.93934166, 66884.303920780934, 0.0, 0.0, 95913693.41127409, -79300.923099238891, 0.0, 0.0, 152734050.03753135, -21615.518484261818, 0.0, 0.0, 230694358.5770939, 22922.170838031663, -383288562.70426351, 0.0, 0.0, 0.0, -126281330.72650661, -26736.867629611937, 0.0, 0.0, 108175824.49057513, -73826.154995947072, 0.0, 0.0, -105130427.45878558, -27109.647812448158, 0.0, 0.0, -495130497.4338429, -43312.494575842124, 0.0, 0.0, -162294563.70339859, -78644.408483291132, 0.0, 0.0, -446143155.07499522, 35049.023211568106, 0.0, 0.0, -434372911.29877228, 74645.281954757884, 0.0, 0.0, 88535885.916778564, -59658.181174639933, 0.0, 0.0, -154524011.19535148, -64925.569280836324, 0.0, 0.0, -197013303.49254256, -57546.998199721042, 0.0, 0.0, 60578806.739350073, 60269.144321032669, 614919978.78591418, 0.0, 0.0, 0.0, -177976472.87455249, 83517.765491276004, 0.0, 0.0, -136301559.45860973, -2057.5912754017359, 0.0, 0.0, 613862755.05429029, -126093.29111144203, 0.0, 0.0, 330986019.80497843, -171293.91106101952, 0.0, 0.0, -29154749.084786706, -23698.75128392793, 0.0, 0.0, 118909415.76059921, 260.4906139722309, 0.0, 0.0, 258338224.08025572, 34588.968913224162, 0.0, 0.0, 3785397.5870792689, -32603.464627771926, 0.0, 0.0, -23259161.770430245, -32648.547687113882, 0.0, 0.0, 104427443.08076131, 5749.4319312391553, 0.0, 0.0, 6766465.5382205453, 3241.742220615246, 0.0, 0.0, -60284503.8862499, -59902.508087108952, 0.0, 0.0, 70947908.65981704, -50046.164234017422, 71592751.406619206, 0.0, 0.0, 0.0, 114423944.11037378, -29629.639271339391, 0.0, 0.0, -16162405.156746721, -22843.554219456128, 0.0, 0.0, -391208214.49301404, 85360.748577972205, 0.0, 0.0, -180559385.64929616, -232405.39276717394, 0.0, 0.0, -173999368.48403537, -154535.5223811192, 0.0, 0.0, 21379759.421723526, -49363.223340975463, 0.0, 0.0, -65362589.496401452, 56497.722354469195, 0.0, 0.0, -59068349.756921723, -22203.228234489961, 0.0, 0.0, 178895590.72285667, -36778.476704805165, 0.0, 0.0, -7977531.617536407, -19838.626327764079, 0.0, 0.0, 80425490.578405693, 27831.980826245854, 0.0, 0.0, 114611568.72714353, -107486.07299270081, 0.0, 0.0, 20196073.66257038, 21928.884663855733, 0.0, 0.0, -77682176.577698827, 29213.006661709769, 0.0, 0.0, 15123630.35168907, -28694.812017467506, -22259183.296553552, 0.0, 0.0, 0.0, -44195359.795093246, 51184.417790776875, 0.0, 0.0, 416191946.56073493, -71732.351641676665, 0.0, 0.0, 205663263.71618012, 3388.5589815610024, 0.0, 0.0, -130943398.42767963, -131143.82698794606, 0.0, 0.0, 61651470.898734599, -81217.981804434457, 0.0, 0.0, -67180046.491523147, -146577.21040483299, 0.0, 0.0, -152936171.75485, -9306.3668560189235, 0.0, 0.0, -78432624.898623183, -16457.284767487032, 0.0, 0.0, -182188345.51379466, -28943.200208655751, 0.0, 0.0, -230936841.81174293, -69881.661699287171, 0.0, 0.0, -48611744.075940624, -45613.670555435718, 0.0, 0.0, -137135254.63530335, 89602.425289011473, 0.0, 0.0, -122086433.73023096, 10038.304046797484, 0.0, 0.0, 4230108.1019680994, -125434.25870475314, 0.0, 0.0, -88554513.106125027, -65172.061009842495, 0.0, 0.0, -48016889.768255182, 6514.6442567450413, 0.0, 0.0, -7234862.3567767777, 29462.421170785103, 23975873.985774957, 0.0, 0.0, 0.0, 150246363.27382952, -4635.0397663657277, 0.0, 0.0, -47057107.317354918, -61032.384893401948, 0.0, 0.0, -145067771.71055323, 30340.435422360377, 0.0, 0.0, 130479661.43167895, -11379.689478798271, 0.0, 0.0, 26361229.859195136, -125845.83502669417, 0.0, 0.0, 169502237.90273514, -190856.12211173368, 0.0, 0.0, -93899694.259581938, 9428.9987870662444, 0.0, 0.0, -197490720.78410134, 21312.280548650102, 0.0, 0.0, 39647940.551049069, -68437.045331174842, 0.0, 0.0, 69474226.516149923, -51031.516066011136, 0.0, 0.0, 183898889.91059858, 25427.755216915164, 0.0, 0.0, 105935367.66732898, 47267.944906588062, 0.0, 0.0, 91688077.289836451, 19266.763736649074, 0.0, 0.0, 107771149.47695087, -67938.256377686776, 0.0, 0.0, -29850724.42625796, 21607.473853242802, 0.0, 0.0, 105178954.14601882, -69775.134273171891, 0.0, 0.0, -57617253.476713084, -65410.799885873028, 0.0, 0.0, 3261331.2317122123, -32217.75104223685, 0.0, 0.0, 16472425.803737164, 79582.269607468625, -272848871.13511133, 0.0, 0.0, 0.0, -166763480.12559777, 33090.611564668841, 0.0, 0.0, 108954734.6009059, -403.73869004175492, 0.0, 0.0, -17283151.399917897, -68588.699644901702, 0.0, 0.0, -21492440.153302398, -17597.541258704674, 0.0, 0.0, -38441136.647693813, 64579.571492712137, 0.0, 0.0, 4505100.6340750242, -300117.69352899096, 0.0, 0.0, 549992.16794250254, -197163.89229418512, 0.0, 0.0, -127963651.33731289, -10644.866941283586, 0.0, 0.0, 100807969.57532065, 29015.182801835355, 0.0, 0.0, -7278488.9224838661, 7477.4234534004445]</td>
      </tr>
      <tr>
        <td id="L220" class="blob-num js-line-number" data-line-number="220"></td>
        <td id="LC220" class="blob-code blob-code-inner js-file-line">        ),</td>
      </tr>
      <tr>
        <td id="L221" class="blob-num js-line-number" data-line-number="221"></td>
        <td id="LC221" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;</span>alm6<span class="pl-pds">&#39;</span></span>: n.array(</td>
      </tr>
      <tr>
        <td id="L222" class="blob-num js-line-number" data-line-number="222"></td>
        <td id="LC222" class="blob-code blob-code-inner js-file-line">[797385716.91804504, 0.0, 0.0, 0.0, 189967106.19923255, 29537.10693528819, 846600538.89322472, 0.0, 0.0, 0.0, 872926192.63071275, 44904.233226988472, 0.0, 0.0, -169861745.64221448, 42971.448932979642, 0.0, 0.0, -147906900.75269288, -39808.417875855441, -1253273886.731957, 0.0, 0.0, 0.0, 7956724.3034045501, -6356.2955966941372, 0.0, 0.0, 699223699.88883495, 19020.68326448747, 0.0, 0.0, 62573658.414382517, 35470.424362177982, 0.0, 0.0, -138953239.49548644, 7299.6975745242362, 0.0, 0.0, -79806306.899763986, 26921.147432152444, 655475892.86221838, 0.0, 0.0, 0.0, -238869250.08572152, 20663.930342020343, 0.0, 0.0, -395488561.2668637, 84491.317123754401, 0.0, 0.0, 426316351.0672881, -45085.033036303044, 0.0, 0.0, 103665104.62868968, 2246.2873997537672, 0.0, 0.0, 264983020.48203912, 1892.1499981893205, 0.0, 0.0, 84976114.845701993, -23012.274489893702, 0.0, 0.0, 50771408.628350258, 27079.015870586507, -174485451.73842844, 0.0, 0.0, 0.0, -264289472.79323781, -56715.616631217796, 0.0, 0.0, -106222730.444833, 74901.88337064792, 0.0, 0.0, -76714164.675549507, 42894.665025885537, 0.0, 0.0, -44453338.477396756, 14708.146540980564, 0.0, 0.0, -127030875.23670675, -23573.026026034317, 0.0, 0.0, -380764603.20410013, -47356.176563799374, 0.0, 0.0, -70038599.590872958, 57096.578562188763, 0.0, 0.0, -110338759.44084603, 15618.47767476493, 0.0, 0.0, -166316476.72216225, -16378.749845538387, 271936085.48576093, 0.0, 0.0, 0.0, 92433839.952875465, 19519.099424088534, 0.0, 0.0, -77618380.050654531, 54389.716587842369, 0.0, 0.0, 77410628.333828956, 18890.257217773775, 0.0, 0.0, 359407589.92209935, 31728.327458281798, 0.0, 0.0, 117813006.27702276, 56694.00896615073, 0.0, 0.0, 322163097.63930309, -24474.159721000109, 0.0, 0.0, 313393639.11823243, -53880.392102794642, 0.0, 0.0, -63859274.184436947, 42560.437368084611, 0.0, 0.0, 111161686.03953889, 46744.631315417995, 0.0, 0.0, 142346033.48720458, 41862.95645676343, 0.0, 0.0, -43818886.848100305, -43470.280716078254, -442768171.11322945, 0.0, 0.0, 0.0, 132501506.90008594, -60540.916997092565, 0.0, 0.0, 93942945.610904098, 2481.2147707751869, 0.0, 0.0, -441000417.33351082, 90293.217816417338, 0.0, 0.0, -237336783.93090552, 124693.18798191204, 0.0, 0.0, 20623105.645757515, 16780.788665236021, 0.0, 0.0, -86673347.981770217, -111.29751227977317, 0.0, 0.0, -187034693.45671436, -25193.745737031823, 0.0, 0.0, -2135530.3974111909, 23393.093986819636, 0.0, 0.0, 16698628.352701169, 22791.910128108695, 0.0, 0.0, -75207242.986817583, -3478.3017686277117, 0.0, 0.0, -4262714.0713599836, -1495.1668203620484, 0.0, 0.0, 43275301.615728691, 43375.651892071954, 0.0, 0.0, -51358762.141454875, 35744.562531746327, -55977125.17363143, 0.0, 0.0, 0.0, -80891874.204180971, 21169.676736598227, 0.0, 0.0, 11039910.066945948, 17564.401170771853, 0.0, 0.0, 282861419.72051185, -61850.098505863178, 0.0, 0.0, 128362573.7168542, 169414.68024107264, 0.0, 0.0, 124330006.22390915, 110718.99879655795, 0.0, 0.0, -16841657.337382279, 35657.177598049217, 0.0, 0.0, 47349371.812364854, -40707.611029805084, 0.0, 0.0, 42010956.049356803, 15687.679769475246, 0.0, 0.0, -129069297.57474856, 25830.387752163271, 0.0, 0.0, 5910511.5135691371, 13697.037347944815, 0.0, 0.0, -58821740.432719804, -19405.588572920107, 0.0, 0.0, -82547653.248470947, 77731.66933892775, 0.0, 0.0, -14433978.003370205, -15837.572457110891, 0.0, 0.0, 55786893.745701335, -21124.411201972158, 0.0, 0.0, -11046375.71224675, 20616.212993154164, 14107144.447629577, 0.0, 0.0, 0.0, 33774805.041091383, -36859.336437719656, 0.0, 0.0, -301070957.00480914, 52878.121995111564, 0.0, 0.0, -146653601.39361849, -2198.0942705195794, 0.0, 0.0, 94168406.841010004, 96760.717803226478, 0.0, 0.0, -43478941.173792779, 57375.658488987028, 0.0, 0.0, 48353110.436083667, 106201.96001161895, 0.0, 0.0, 112485321.00221781, 6382.1773339061747, 0.0, 0.0, 57189418.603424326, 11901.926954900946, 0.0, 0.0, 131280556.01794536, 20435.530265815629, 0.0, 0.0, 166170342.25571772, 50100.982288524996, 0.0, 0.0, 35409830.459971406, 33124.972929883166, 0.0, 0.0, 98747876.406133488, -63736.080320002809, 0.0, 0.0, 88071880.266417414, -7100.2184152616755, 0.0, 0.0, -3067340.9423153657, 91162.549369077489, 0.0, 0.0, 63897911.553277262, 46684.560473517355, 0.0, 0.0, 34498206.032984555, -4537.3240835134584, 0.0, 0.0, 5168843.1048901472, -21320.40983260217, -15835543.717267705, 0.0, 0.0, 0.0, -108625585.50132097, 3294.6252010393659, 0.0, 0.0, 33621479.273489162, 44729.560439200985, 0.0, 0.0, 103347563.63279098, -21767.597065975646, 0.0, 0.0, -94941255.180099696, 9685.7584460242178, 0.0, 0.0, -18528876.37688531, 90099.006466248247, 0.0, 0.0, -121579532.1146235, 139213.3284709499, 0.0, 0.0, 71246149.768020689, -6806.3276870759018, 0.0, 0.0, 145604934.16640645, -15090.6173931348, 0.0, 0.0, -29251884.513770111, 49390.368939568521, 0.0, 0.0, -49586552.469104417, 36886.524239895676, 0.0, 0.0, -132737515.14220525, -18372.148025799193, 0.0, 0.0, -76491981.799765065, -33802.178523338815, 0.0, 0.0, -65923648.161891818, -13764.57393849405, 0.0, 0.0, -77827781.713095531, 47692.036396206728, 0.0, 0.0, 21647589.21944366, -14133.730494852018, 0.0, 0.0, -76290028.632632598, 51182.252677407232, 0.0, 0.0, 41455286.640109941, 47157.905514409278, 0.0, 0.0, -2303036.9134609378, 23541.969363905846, 0.0, 0.0, -11918806.087500827, -57397.776327919324, 197612684.36048445, 0.0, 0.0, 0.0, 118955416.99244839, -24087.627570760262, 0.0, 0.0, -77491887.254076943, 728.30824844376821, 0.0, 0.0, 12207030.001704402, 49466.405483031638, 0.0, 0.0, 14932883.729976833, 14239.9646008034, 0.0, 0.0, 27192582.807980306, -46905.646869159398, 0.0, 0.0, -3736072.351231684, 218791.39737395261, 0.0, 0.0, 938468.7842925014, 140973.07684057893, 0.0, 0.0, 94550054.572198883, 7941.4520718544645, 0.0, 0.0, -74181966.616232648, -21044.559100278962, 0.0, 0.0, 5019241.6634857096, -5304.694057288676]</td>
      </tr>
      <tr>
        <td id="L223" class="blob-num js-line-number" data-line-number="223"></td>
        <td id="LC223" class="blob-code blob-code-inner js-file-line">        ),</td>
      </tr>
      <tr>
        <td id="L224" class="blob-num js-line-number" data-line-number="224"></td>
        <td id="LC224" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;</span>alm5<span class="pl-pds">&#39;</span></span>: n.array(</td>
      </tr>
      <tr>
        <td id="L225" class="blob-num js-line-number" data-line-number="225"></td>
        <td id="LC225" class="blob-code blob-code-inner js-file-line">[-197735155.01929793, 0.0, 0.0, 0.0, -46053418.11639905, -7226.397802963189, -210520973.5427039, 0.0, 0.0, 0.0, -210657534.4115783, -10745.785564600596, 0.0, 0.0, 41148658.926344283, -10416.636060571493, 0.0, 0.0, 35784980.360108547, 9612.9840553492686, 304168753.30395395, 0.0, 0.0, 0.0, -3117586.5713760527, 1632.8412811888347, 0.0, 0.0, -171090196.95194718, -4632.55322212383, 0.0, 0.0, -15011356.299438834, -8364.6158748977232, 0.0, 0.0, 33676784.554911196, -1772.7269020579031, 0.0, 0.0, 19210024.26669988, -6481.5616124462031, -158387393.28245038, 0.0, 0.0, 0.0, 57914671.611356676, -4885.2721544610504, 0.0, 0.0, 95684065.272677362, -20637.725554880592, 0.0, 0.0, -102657366.72310401, 10831.674992370326, 0.0, 0.0, -24919568.123357572, -493.6485726144798, 0.0, 0.0, -63605529.998098962, -674.09128358241719, 0.0, 0.0, -20267019.847643819, 5547.5212175440292, 0.0, 0.0, -12278347.932371413, -6453.4324016264673, 43450364.563942656, 0.0, 0.0, 0.0, 64756329.325183183, 13568.438404067152, 0.0, 0.0, 25774658.376369122, -18443.874764878143, 0.0, 0.0, 18490347.059644662, -10189.922507207644, 0.0, 0.0, 11237512.512576636, -3548.1511247369526, 0.0, 0.0, 30643195.288836252, 5554.058991234795, 0.0, 0.0, 91852213.821721107, 11152.354180029373, 0.0, 0.0, 17136791.711244829, -13738.340482599324, 0.0, 0.0, 26654802.354939841, -3772.9607028607643, 0.0, 0.0, 40080801.777484469, 3898.1337497247532, -64205745.592204437, 0.0, 0.0, 0.0, -22659883.342662893, -4778.0834184917312, 0.0, 0.0, 18555022.616979484, -13473.030524613543, 0.0, 0.0, -19158703.781450976, -4325.618633195475, 0.0, 0.0, -87303680.674776122, -7798.5650239280103, 0.0, 0.0, -28630028.187836513, -13657.289990676993, 0.0, 0.0, -77776433.152057096, 5663.1889376592408, 0.0, 0.0, -75576903.129559204, 12998.603107606816, 0.0, 0.0, 15394598.358521122, -10123.752482296954, 0.0, 0.0, -26711379.163051497, -11242.177700866312, 0.0, 0.0, -34392594.777521037, -10202.774681930312, 0.0, 0.0, 10602956.226330029, 10479.795259434368, 106491522.00151923, 0.0, 0.0, 0.0, -33138240.729237858, 14693.306846559934, 0.0, 0.0, -21364417.055660442, -922.28831526417491, 0.0, 0.0, 105835946.52125469, -21543.372550449742, 0.0, 0.0, 56811774.725853391, -30413.430161657085, 0.0, 0.0, -4861543.3597652912, -3938.4477312154736, 0.0, 0.0, 21178757.320129372, 3.5425208576304499, 0.0, 0.0, 45298346.835260376, 6143.2888814262978, 0.0, 0.0, 335546.7987198591, -5597.459067079385, 0.0, 0.0, -3999171.3806912508, -5264.6744997915521, 0.0, 0.0, 18095332.294255983, 638.8459776001049, 0.0, 0.0, 840929.63493319822, 107.15833138602918, 0.0, 0.0, -10371226.677820578, -10505.444300287199, 0.0, 0.0, 12436856.257415403, -8511.9205882707138, 14784850.940883569, 0.0, 0.0, 0.0, 19007856.207817975, -5034.9923737995823, 0.0, 0.0, -2498879.0170426308, -4590.8614967715039, 0.0, 0.0, -68376948.624012321, 15000.851884991811, 0.0, 0.0, -30424090.555139855, -41397.471186152834, 0.0, 0.0, -29616739.442200258, -26436.200386076027, 0.0, 0.0, 4504589.81844257, -8603.13572616479, 0.0, 0.0, -11478082.127457535, 9801.1835365541247, 0.0, 0.0, -9949835.1498631909, -3681.9065933991928, 0.0, 0.0, 31127145.710026521, -6023.8911849642827, 0.0, 0.0, -1474616.7280308811, -3120.8527051806514, 0.0, 0.0, 14424342.151320595, 4478.0547817086999, 0.0, 0.0, 19870777.76114646, -18798.894308055569, 0.0, 0.0, 3439581.5227250052, 3824.0816135823075, 0.0, 0.0, -13375210.583126174, 5111.8038311824457, 0.0, 0.0, 2704300.4055295889, -4943.7362257902059, -2820776.3819985995, 0.0, 0.0, 0.0, -8716712.5975790638, 8870.4368321870552, 0.0, 0.0, 72838691.512315914, -13115.675171949872, 0.0, 0.0, 34868715.134384163, 445.01372842380943, 0.0, 0.0, -22648556.756227445, -24007.562182239992, 0.0, 0.0, 10225496.43410253, -13429.948589291756, 0.0, 0.0, -11648114.4951357, -25752.408984327056, 0.0, 0.0, -27790363.240071055, -1440.4752597467768, 0.0, 0.0, -13972388.61876422, -2880.3368818686808, 0.0, 0.0, -31611811.476835247, -4796.544240381837, 0.0, 0.0, -39940933.139877774, -11990.248686645078, 0.0, 0.0, -8637664.214980036, -8054.465313268588, 0.0, 0.0, -23757834.173265617, 15100.165734198441, 0.0, 0.0, -21240042.369048979, 1670.9006386179863, 0.0, 0.0, 745992.55951904377, -22177.201914765799, 0.0, 0.0, -15411608.939690793, -11155.870022900486, 0.0, 0.0, -8277784.326783523, 1045.3560798711144, 0.0, 0.0, -1230781.1765813681, 5157.2676594239729, 3371226.6051944932, 0.0, 0.0, 0.0, 26258888.926003013, -781.11690680106267, 0.0, 0.0, -7982799.094995562, -11012.550602180869, 0.0, 0.0, -24555994.147887498, 5209.5952756182242, 0.0, 0.0, 23146703.87589569, -2812.6513443658323, 0.0, 0.0, 4318014.4979850184, -21492.21656303855, 0.0, 0.0, 29058640.345119234, -34037.549503927024, 0.0, 0.0, -18242673.078121372, 1656.8324467507082, 0.0, 0.0, -36072862.830247574, 3551.0799378540246, 0.0, 0.0, 7252934.6827932158, -11913.82448328409, 0.0, 0.0, 11797958.62110701, -8915.0206777991007, 0.0, 0.0, 32028437.321352843, 4440.2513551878992, 0.0, 0.0, 18463453.022954959, 8064.832811994861, 0.0, 0.0, 15830357.645304311, 3278.6002973079399, 0.0, 0.0, 18792715.22209141, -11110.350620979698, 0.0, 0.0, -5254231.7428235617, 2977.060123896787, 0.0, 0.0, 18517061.004637863, -12596.961454058939, 0.0, 0.0, -9965178.2132176459, -11359.992935143997, 0.0, 0.0, 541776.1522784296, -5769.2477330948705, 0.0, 0.0, 2885107.8830083506, 13837.858540130937, -47862763.656404138, 0.0, 0.0, 0.0, -28298286.980327792, 5873.9680203106927, 0.0, 0.0, 18362400.235881787, -315.51705578927215, 0.0, 0.0, -2874422.5857720273, -11929.116229145047, 0.0, 0.0, -3417477.3191902661, -3924.658545164405, 0.0, 0.0, -6421891.4478960913, 11416.933954008207, 0.0, 0.0, 1037940.6079158594, -53486.786972605179, 0.0, 0.0, -613407.33829156274, -33589.737764611818, 0.0, 0.0, -23490363.036867645, -2004.1753650588878, 0.0, 0.0, 18333266.803092051, 5109.7982999645592, 0.0, 0.0, -1138365.5078694951, 1253.4638702863897]</td>
      </tr>
      <tr>
        <td id="L226" class="blob-num js-line-number" data-line-number="226"></td>
        <td id="LC226" class="blob-code blob-code-inner js-file-line">        ),</td>
      </tr>
      <tr>
        <td id="L227" class="blob-num js-line-number" data-line-number="227"></td>
        <td id="LC227" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;</span>alm4<span class="pl-pds">&#39;</span></span>: n.array(</td>
      </tr>
      <tr>
        <td id="L228" class="blob-num js-line-number" data-line-number="228"></td>
        <td id="LC228" class="blob-code blob-code-inner js-file-line">[7200724.2174806157, 0.0, 0.0, 0.0, 1507578.5364308401, 249.09445266642169, 7725779.5666901879, 0.0, 0.0, 0.0, 6747539.2579509569, 329.36641952671226, 0.0, 0.0, -1343001.5699128646, 343.36792991568888, 0.0, 0.0, -1159855.5653791095, -308.92769367092768, -10045128.444344388, 0.0, 0.0, 0.0, 288589.31376132951, -70.980762890053413, 0.0, 0.0, 5851186.7774917074, 155.21204435663131, 0.0, 0.0, 467714.86876319919, 238.59930858441089, 0.0, 0.0, -1101842.335750744, 58.096309466026973, 0.0, 0.0, -608138.07580545719, 204.95821814484245, 5101024.1326503837, 0.0, 0.0, 0.0, -1885231.9571066336, 139.59983322242354, 0.0, 0.0, -3105913.7215474164, 701.68124144120748, 0.0, 0.0, 3239644.93434321, -339.82491448090769, 0.0, 0.0, 783855.81893535459, 8.7527052982549058, 0.0, 0.0, 1986537.8514804086, 54.774986424606197, 0.0, 0.0, 612170.88910865143, -176.99708425924547, 0.0, 0.0, 398007.89088467637, 193.56614271468877, -1596925.8860192283, 0.0, 0.0, 0.0, -2227878.7078425568, -415.01509470166883, 0.0, 0.0, -844775.91825557721, 650.38469558451084, 0.0, 0.0, -590632.35791223042, 299.36217992006749, 0.0, 0.0, -437078.74958942051, 113.79355867828019, 0.0, 0.0, -980335.68394004204, -157.54883485169805, 0.0, 0.0, -2938603.615868886, -315.29147746550342, 0.0, 0.0, -584687.47979809716, 434.71263424168563, 0.0, 0.0, -858991.85386585491, 121.46105830492419, 0.0, 0.0, -1277548.8225962832, -115.89194500169798, 1840668.8290774515, 0.0, 0.0, 0.0, 775421.56337957992, 164.00611331607877, 0.0, 0.0, -562088.42813265021, 488.74685971765177, 0.0, 0.0, 691778.87208729645, 97.689901957009482, 0.0, 0.0, 2882651.1565228878, 272.52207724336955, 0.0, 0.0, 948662.06765682506, 433.74488550190046, 0.0, 0.0, 2497867.2332102088, -144.98233385054755, 0.0, 0.0, 2414321.2649437124, -415.58171649203098, 0.0, 0.0, -490977.9606196385, 302.58764598967997, 0.0, 0.0, 838801.66186270467, 354.24491208439002, 0.0, 0.0, 1109240.4625873815, 343.14609143330898, 0.0, 0.0, -344528.03137613513, -334.26528674880853, -3354305.0711071114, 0.0, 0.0, 0.0, 1235458.4567361525, -485.19168633800058, 0.0, 0.0, 487397.2070375224, 83.880871552662413, 0.0, 0.0, -3308936.6858345182, 647.62704156274015, 0.0, 0.0, -1751740.8660837077, 1026.5527017383265, 0.0, 0.0, 139846.54955892122, 106.93161870083593, 0.0, 0.0, -721404.82398454682, 3.6015241937210991, 0.0, 0.0, -1476829.0591856656, -205.98964494497451, 0.0, 0.0, 17074.077977673544, 171.31880268277882, 0.0, 0.0, 123060.47869451297, 132.0062185627576, 0.0, 0.0, -571603.6917338561, 10.396528401168975, 0.0, 0.0, 2106.5454300025908, 35.752699013801632, 0.0, 0.0, 321464.4592717489, 342.34076722166759, 0.0, 0.0, -405230.5941204023, 255.2360120010726, -670296.34645801655, 0.0, 0.0, 0.0, -529100.52793346881, 148.57741762277331, 0.0, 0.0, 57471.51362526336, 206.86269211261629, 0.0, 0.0, 2207181.4600743535, -493.51422675565459, 0.0, 0.0, 895463.30568390526, 1409.934585353082, 0.0, 0.0, 888776.11402601667, 798.19028910433144, 0.0, 0.0, -215643.35608039363, 274.1110708271608, 0.0, 0.0, 376051.83866420202, -310.73247764113745, 0.0, 0.0, 289955.29165051226, 101.76182819162518, 0.0, 0.0, -994834.62761617475, 161.69117608964245, 0.0, 0.0, 55076.980514253126, 72.037278617768209, 0.0, 0.0, -497694.41516721481, -112.23417665406721, 0.0, 0.0, -630133.16098474909, 608.3926186109635, 0.0, 0.0, -103502.07506563514, -122.78257869786432, 0.0, 0.0, 415209.12744939065, -166.58248247027083, 0.0, 0.0, -92606.301129069601, 153.27094286810689, -155.133754694667, 0.0, 0.0, 0.0, 368304.00212202733, -280.867215040259, 0.0, 0.0, -2362905.2792973067, 479.92664306566542, 0.0, 0.0, -1040034.2467855478, 0.76582903775922517, 0.0, 0.0, 718560.8191561898, 876.24115424409968, 0.0, 0.0, -292721.47833479074, 359.34134632343324, 0.0, 0.0, 372865.53641371639, 845.74282823210956, 0.0, 0.0, 993400.71737590164, 30.865235806642321, 0.0, 0.0, 474280.45441336668, 93.872153111048078, 0.0, 0.0, 1002907.9299558332, 133.19243145625578, 0.0, 0.0, 1255885.0030796861, 369.38663831551793, 0.0, 0.0, 290708.02660839848, 267.74414177703102, 0.0, 0.0, 750690.63702933991, -441.17311505823324, 0.0, 0.0, 679277.21542988182, -47.164992914979038, 0.0, 0.0, -25034.048713920944, 737.20036885145737, 0.0, 0.0, 492703.18828978809, 340.11997934964194, 0.0, 0.0, 258268.03397987972, -25.875498151930167, 0.0, 0.0, 36841.386016291726, -166.58153344592151, -37632.524860587197, 0.0, 0.0, 0.0, -846970.57810843072, 23.067138772739717, 0.0, 0.0, 231772.05047900774, 389.59227512856796, 0.0, 0.0, 731573.15687432117, -159.47722701117917, 0.0, 0.0, -779636.79196457902, 169.42221635624287, 0.0, 0.0, -114048.4415545105, 645.30053706908575, 0.0, 0.0, -881108.21860260586, 1161.5099955044125, 0.0, 0.0, 748235.23576389183, -57.690368801478542, 0.0, 0.0, 1304114.5518882216, -99.391608662850047, 0.0, 0.0, -263144.21203343244, 380.80042850864618, 0.0, 0.0, -352403.58508126321, 287.70856589182199, 0.0, 0.0, -1026027.1112983709, -143.58640740725582, 0.0, 0.0, -592252.06919899839, -244.84725404024846, 0.0, 0.0, -495445.89352101792, -98.364381497365002, 0.0, 0.0, -604362.59029897652, 295.72251912004111, 0.0, 0.0, 173141.05309468889, -29.437403866957482, 0.0, 0.0, -609597.44962526858, 441.85413291552464, 0.0, 0.0, 313734.85551236029, 360.67490478600058, 0.0, 0.0, -15401.76767474161, 198.95971188667156, 0.0, 0.0, -93905.143248108579, -441.64480338592182, 1557834.4870206926, 0.0, 0.0, 0.0, 846825.25105827861, -197.83739354926948, 0.0, 0.0, -537733.70476057276, 33.21990588911023, 0.0, 0.0, 82182.540697250195, 381.92187491491592, 0.0, 0.0, 79199.852686637649, 206.0957338460905, 0.0, 0.0, 188168.68771215889, -382.92038636172128, 0.0, 0.0, -53095.127393283889, 1828.4815368321251, 0.0, 0.0, 77039.429637870722, 1006.0372475571642, 0.0, 0.0, 860002.40107733512, 79.645120143879822, 0.0, 0.0, -655526.16343116295, -168.77744731668088, 0.0, 0.0, 24864.210627988363, -36.250655713179249]</td>
      </tr>
      <tr>
        <td id="L229" class="blob-num js-line-number" data-line-number="229"></td>
        <td id="LC229" class="blob-code blob-code-inner js-file-line">        ),</td>
      </tr>
      <tr>
        <td id="L230" class="blob-num js-line-number" data-line-number="230"></td>
        <td id="LC230" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;</span>alm3<span class="pl-pds">&#39;</span></span>: n.array(</td>
      </tr>
      <tr>
        <td id="L231" class="blob-num js-line-number" data-line-number="231"></td>
        <td id="LC231" class="blob-code blob-code-inner js-file-line">[5498273.867212004, 0.0, 0.0, 0.0, 1278787.6144197686, 200.69631150016968, 5855938.7480396256, 0.0, 0.0, 0.0, 5847379.6897321241, 298.14002897401355, 0.0, 0.0, -1142519.2128262592, 289.14639793614521, 0.0, 0.0, -993520.89184620359, -266.85901085140711, -8445890.7411452159, 0.0, 0.0, 0.0, 88960.996282732958, -45.404243903980827, 0.0, 0.0, 4753726.6035505533, 128.65976222572772, 0.0, 0.0, 416486.45794872148, 231.75264200679231, 0.0, 0.0, -935084.04088520445, 49.245386113225521, 0.0, 0.0, -533118.40524846269, 179.90326954511633, 4396953.9195928695, 0.0, 0.0, 0.0, -1608349.7299925357, 135.45560067565393, 0.0, 0.0, -2656511.2093055765, 573.27704161730151, 0.0, 0.0, 2849651.3732845969, -300.56300083744014, 0.0, 0.0, 691500.88130568538, 13.583711800361833, 0.0, 0.0, 1764810.4259444436, 19.179618441300519, 0.0, 0.0, 562066.86523355253, -153.96198960788544, 0.0, 0.0, 340862.09253124218, 178.98642747530579, -1208920.608357993, 0.0, 0.0, 0.0, -1799262.1597735337, -376.45318693479493, 0.0, 0.0, -715874.06723833689, 512.60274437112969, 0.0, 0.0, -513195.85187474889, 282.61204774423248, 0.0, 0.0, -312987.8115696148, 98.476599955903495, 0.0, 0.0, -850554.04150807834, -153.88002183190986, 0.0, 0.0, -2549521.2039438705, -308.97811247523509, 0.0, 0.0, -476185.0271458606, 381.24956233215812, 0.0, 0.0, -739921.59786708606, 104.73864708094125, 0.0, 0.0, -1112406.7067007029, -108.11473746144854, 1779181.7167289439, 0.0, 0.0, 0.0, 629948.71285196778, 132.75183050912085, 0.0, 0.0, -514848.34386742389, 374.55990412960182, 0.0, 0.0, 532627.3153818252, 119.77835605693443, 0.0, 0.0, 2424622.4174201251, 216.72201228006872, 0.0, 0.0, 795108.48625639779, 379.06163904571338, 0.0, 0.0, 2158946.4330264404, -156.65280944607056, 0.0, 0.0, 2097723.367115201, -360.82043797325053, 0.0, 0.0, -427280.22268814908, 280.67854240034717, 0.0, 0.0, 741194.31265485799, 311.98234210405832, 0.0, 0.0, 954730.11413698632, 283.38882535844346, 0.0, 0.0, -294365.82031801599, -290.87943522870074, -2955385.5772971096, 0.0, 0.0, 0.0, 922348.17132076842, -407.96866449942939, 0.0, 0.0, 590219.54455448966, 26.085003675834287, 0.0, 0.0, -2936385.3988425806, 597.68610206583253, 0.0, 0.0, -1575950.5946588507, 844.75891669718953, 0.0, 0.0, 134649.35790006144, 109.16816207238172, 0.0, 0.0, -588332.72589620715, -0.060434616306956457, 0.0, 0.0, -1257716.4221816759, -170.68406812560957, 0.0, 0.0, -8941.0779337873792, 155.29124403630357, 0.0, 0.0, 110950.05042145777, 145.66225385101623, 0.0, 0.0, -502170.41979421745, -17.325868884214849, 0.0, 0.0, -22949.505096291126, -2.4551467208407827, 0.0, 0.0, 287724.38079964806, 291.68391922333836, 0.0, 0.0, -345306.35353751393, 236.04471625211903, -412926.76103576075, 0.0, 0.0, 0.0, -526646.50936312054, 139.67052359284023, 0.0, 0.0, 68989.951308911957, 127.92940529824081, 0.0, 0.0, 1898231.0764858185, -416.49200396660581, 0.0, 0.0, 843251.14376973349, 1149.9715407003903, 0.0, 0.0, 821330.99892625911, 733.43891047406044, 0.0, 0.0, -125797.69610716954, 238.84073917328871, 0.0, 0.0, 318711.61841675505, -272.01264234407392, 0.0, 0.0, 275787.48754996213, 101.98949961733558, 0.0, 0.0, -863964.31932292669, 166.75001032516667, 0.0, 0.0, 41026.459186118453, 86.226676317826517, 0.0, 0.0, -400863.91227860982, -123.85056930956991, 0.0, 0.0, -551443.79119247897, 521.92551449632617, 0.0, 0.0, -95383.356716058464, -106.16353997548579, 0.0, 0.0, 371079.83600833849, -141.89502251030711, 0.0, 0.0, -75145.763091870656, 137.17426779366033, 77215.485549988181, 0.0, 0.0, 0.0, 243010.03290878746, -246.16028356941868, 0.0, 0.0, -2022146.4903465142, 364.58173021633223, 0.0, 0.0, -966729.65752818203, -12.242870088295632, 0.0, 0.0, 628405.85512085771, 667.51076371889519, 0.0, 0.0, -283104.73741182394, 372.19399604828794, 0.0, 0.0, 323261.32509660453, 715.00664543583957, 0.0, 0.0, 772628.27728037443, 39.783619682754797, 0.0, 0.0, 388194.79034767463, 79.95535765348265, 0.0, 0.0, 877314.47408925625, 132.86186416546855, 0.0, 0.0, 1108321.017616556, 332.59517110546255, 0.0, 0.0, 239961.36277227409, 223.69078367662058, 0.0, 0.0, 659299.35428607883, -418.54744902859727, 0.0, 0.0, 589532.25215103338, -46.289537699833957, 0.0, 0.0, -20714.541405546493, 615.98609025910253, 0.0, 0.0, 427769.59997919854, 309.46495603679875, 0.0, 0.0, 229667.51363210069, -28.911630023187641, 0.0, 0.0, 34129.567905943033, -143.20187626491094, -92640.48420559497, 0.0, 0.0, 0.0, -729109.33367109543, 21.654587090135621, 0.0, 0.0, 221416.5410305266, 306.01150060926915, 0.0, 0.0, 680747.43978198699, -144.57062058568377, 0.0, 0.0, -642943.0025409261, 78.812483102076229, 0.0, 0.0, -119574.15208283922, 596.2647968341123, 0.0, 0.0, -806204.14766925236, 945.57520214706699, 0.0, 0.0, 508496.6306797663, -45.935683181243427, 0.0, 0.0, 1003116.1598382751, -98.403721000389453, 0.0, 0.0, -201696.11249167149, 330.7020816302512, 0.0, 0.0, -327125.10138588975, 247.48897983897257, 0.0, 0.0, -889021.40818739205, -123.25802427575343, 0.0, 0.0, -512511.74050718872, -223.64554240580921, 0.0, 0.0, -439243.26699259342, -90.926884091497826, 0.0, 0.0, -521657.70312392223, 307.52889167500388, 0.0, 0.0, 145905.88035071047, -81.706295272192676, 0.0, 0.0, -514217.75957826863, 350.16556747675372, 0.0, 0.0, 276521.46974886092, 315.29551730855343, 0.0, 0.0, -15005.362914721836, 160.30373127823759, 0.0, 0.0, -80100.914586450817, -384.07908162118889, 1329022.0543307522, 0.0, 0.0, 0.0, 784583.09379498218, -163.16917368552592, 0.0, 0.0, -508966.00275964983, 8.9839949262181413, 0.0, 0.0, 79637.436419199701, 331.08436496876726, 0.0, 0.0, 94517.825357591864, 109.72964748155799, 0.0, 0.0, 177860.45162952511, -317.03718781317883, 0.0, 0.0, -29122.396674515963, 1485.7265036016379, 0.0, 0.0, 17912.116468879263, 931.65335058043877, 0.0, 0.0, 653330.83172917052, 55.731087572455294, 0.0, 0.0, -509734.47426063882, -141.90121461573341, 0.0, 0.0, 31458.886496181502, -34.732735905242301]</td>
      </tr>
      <tr>
        <td id="L232" class="blob-num js-line-number" data-line-number="232"></td>
        <td id="LC232" class="blob-code blob-code-inner js-file-line">        ),</td>
      </tr>
      <tr>
        <td id="L233" class="blob-num js-line-number" data-line-number="233"></td>
        <td id="LC233" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;</span>alm2<span class="pl-pds">&#39;</span></span>: n.array(</td>
      </tr>
      <tr>
        <td id="L234" class="blob-num js-line-number" data-line-number="234"></td>
        <td id="LC234" class="blob-code blob-code-inner js-file-line">[-1103414.2044201535, 0.0, 0.0, 0.0, -249216.4175125043, -39.729408989111761, -1176577.5024123464, 0.0, 0.0, 0.0, -1133438.0896773722, -57.136484321740362, 0.0, 0.0, 222514.96389323997, -56.54701176052766, 0.0, 0.0, 193125.4799435687, 51.765853432263377, 1650813.1434357744, 0.0, 0.0, 0.0, -25180.167167042251, 9.706549888877543, 0.0, 0.0, -937103.70299756597, -25.235430095798019, 0.0, 0.0, -80206.951061589556, -43.737009902656311, 0.0, 0.0, 182236.99738128181, -9.5841647270127233, 0.0, 0.0, 103053.30746878275, -34.746107248883426, -853322.86337260832, 0.0, 0.0, 0.0, 312646.94046925154, -25.503097047283237, 0.0, 0.0, 516702.80610491917, -112.8995326796007, 0.0, 0.0, -549784.93980406516, 57.973309366387355, 0.0, 0.0, -133464.03856328354, -2.3579576847100228, 0.0, 0.0, -340036.97046563891, -5.0761598513730801, 0.0, 0.0, -107417.89539361646, 29.82477180699183, 0.0, 0.0, -66288.742088489264, -34.12677027245843, 242761.66286661208, 0.0, 0.0, 0.0, 355299.14864236064, 72.099452268713861, 0.0, 0.0, 139443.07563040542, -101.94519349600016, 0.0, 0.0, 99457.074090162219, -53.571788791621884, 0.0, 0.0, 63825.321563384627, -19.106242946089228, 0.0, 0.0, 164839.16754470853, 28.99853104429242, 0.0, 0.0, 494099.56746437453, 58.175733468573178, 0.0, 0.0, 93765.53513571863, -73.705546017026805, 0.0, 0.0, 143668.77177549744, -20.327210721837712, 0.0, 0.0, 215432.54358751758, 20.561875291271971, -336035.2869258151, 0.0, 0.0, 0.0, -123952.11552198855, -26.20554895407496, 0.0, 0.0, 98272.253111928076, -75.114433478302843, 0.0, 0.0, -106650.06951936896, -21.330579531982103, 0.0, 0.0, -473476.94996891997, -42.98730216491024, 0.0, 0.0, -155456.52564638408, -73.316932421345186, 0.0, 0.0, -418808.97740108712, 28.899145302574603, 0.0, 0.0, -406392.02990293864, 69.897831029364852, 0.0, 0.0, 82745.855904357944, -53.543664420697027, 0.0, 0.0, -143003.38259869698, -60.231381402686893, 0.0, 0.0, -185414.39992036307, -55.638999723792324, 0.0, 0.0, 57277.617390907042, 56.327787434392484, 570416.63114454073, 0.0, 0.0, 0.0, -185825.17149148285, 79.745406813858992, 0.0, 0.0, -106401.39466975507, -7.4871873591579607, 0.0, 0.0, 566118.44264289935, -113.91061252917012, 0.0, 0.0, 302767.57112975762, -166.02607098229137, 0.0, 0.0, -25511.390371765974, -20.284633946514809, 0.0, 0.0, 115918.21036339784, -0.14909046442531992, 0.0, 0.0, 244883.77982795733, 33.441292469906351, 0.0, 0.0, 578.97737984670971, -29.749816407437436, 0.0, 0.0, -21287.86151421025, -26.705101674036737, 0.0, 0.0, 97016.665438425509, 2.0789083598333034, 0.0, 0.0, 3245.3467862179295, -1.1507130817224245, 0.0, 0.0, -55339.697968340421, -56.781836595897254, 0.0, 0.0, 67224.568837156505, -45.034112413907174, 88228.413091622468, 0.0, 0.0, 0.0, 98729.870790065805, -26.490575176241236, 0.0, 0.0, -12521.061489840857, -27.498884777435311, 0.0, 0.0, -368630.92316534434, 81.327104401357062, 0.0, 0.0, -160372.31410858282, -226.57241037930862, 0.0, 0.0, -156704.90196554124, -139.95065504138012, 0.0, 0.0, 27482.694924666503, -46.191163980334068, 0.0, 0.0, -62136.927476522571, 52.599397698604172, 0.0, 0.0, -52277.755677365676, -19.094727148594099, 0.0, 0.0, 167405.17967058913, -31.063717548123311, 0.0, 0.0, -8289.0844077842721, -15.577974192081774, 0.0, 0.0, 79186.162530883637, 22.734818682221846, 0.0, 0.0, 106668.34326342445, -101.41692049139429, 0.0, 0.0, 18214.641074740575, 20.585373531073557, 0.0, 0.0, -71390.592043344805, 27.649348022334909, 0.0, 0.0, 14814.590103047942, -26.36849999764064, -11157.116779309097, 0.0, 0.0, 0.0, -50884.637809017622, 47.597477185206181, 0.0, 0.0, 393267.6756285142, -73.354139067332838, 0.0, 0.0, 184321.32496451217, 1.6799736455904748, 0.0, 0.0, -121684.25286297864, -134.07023722185465, 0.0, 0.0, 53659.100234631303, -68.944001271453899, 0.0, 0.0, -62729.732767350608, -139.53389739397599, 0.0, 0.0, -154172.64228266297, -7.0791581577233611, 0.0, 0.0, -76362.086452639225, -15.578970227018429, 0.0, 0.0, -169687.45011097338, -24.916016619164235, 0.0, 0.0, -213898.37319506073, -63.888596650200135, 0.0, 0.0, -47086.22333360646, -43.77647795152663, 0.0, 0.0, -127397.09602717002, 79.396693346221724, 0.0, 0.0, -114260.87383607897, 8.715503344160485, 0.0, 0.0, 4070.2158591931084, -120.48808286434839, 0.0, 0.0, -82888.947500125505, -59.255380108883756, 0.0, 0.0, -44246.638222694237, 5.2910059817257249, 0.0, 0.0, -6508.0061320476525, 27.794706552026138, 15023.702290156509, 0.0, 0.0, 0.0, 141495.08754248972, -4.1209650818827823, 0.0, 0.0, -41796.81177522293, -60.968327854584736, 0.0, 0.0, -129810.40849011859, 27.679103738189752, 0.0, 0.0, 126273.89360457758, -18.804328613744513, 0.0, 0.0, 22145.352034173586, -113.60303290931722, 0.0, 0.0, 153991.20384008222, -186.36981064621975, 0.0, 0.0, -105424.03359466669, 9.1868043948482665, 0.0, 0.0, -200753.94837883816, 18.462385825155181, 0.0, 0.0, 40407.548132653654, -64.063903113355181, 0.0, 0.0, 62369.105084695089, -48.063478622828697, 0.0, 0.0, 172352.2462180329, 23.959278790922735, 0.0, 0.0, 99384.432937844947, 42.821236641167104, 0.0, 0.0, 84676.398507017133, 17.345700727270732, 0.0, 0.0, 101237.19550751503, -57.169603738392134, 0.0, 0.0, -28488.049510846889, 13.144012068357325, 0.0, 0.0, 100356.14379977378, -69.478218605370571, 0.0, 0.0, -53390.779412722877, -60.97825600786021, 0.0, 0.0, 2834.413225208727, -31.681493659170407, 0.0, 0.0, 15592.097028410077, 74.394985075632434, -258548.41923429139, 0.0, 0.0, 0.0, -149722.17678243664, 32.038106652580687, 0.0, 0.0, 96622.532589511451, -2.7613786699687144, 0.0, 0.0, -15055.185742227035, -64.205006666819855, 0.0, 0.0, -17013.860784224202, -24.8088027258508, 0.0, 0.0, -33891.222331880599, 62.248672839967249, 0.0, 0.0, 6427.1304293357562, -293.13055304372818, 0.0, 0.0, -5760.4541098594109, -177.54384831356066, 0.0, 0.0, -131226.58104731468, -11.521255308232114, 0.0, 0.0, 101695.66103538357, 27.724077681902834, 0.0, 0.0, -5604.4363708451438, 6.5783684475226076]</td>
      </tr>
      <tr>
        <td id="L235" class="blob-num js-line-number" data-line-number="235"></td>
        <td id="LC235" class="blob-code blob-code-inner js-file-line">        ),</td>
      </tr>
      <tr>
        <td id="L236" class="blob-num js-line-number" data-line-number="236"></td>
        <td id="LC236" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;</span>alm1<span class="pl-pds">&#39;</span></span>: n.array(</td>
      </tr>
      <tr>
        <td id="L237" class="blob-num js-line-number" data-line-number="237"></td>
        <td id="LC237" class="blob-code blob-code-inner js-file-line">[86105.710512555146, 0.0, 0.0, 0.0, 19016.380367371166, 3.0740357044470707, 91816.647180154585, 0.0, 0.0, 0.0, 86145.045560129452, 4.3020547568553216, 0.0, 0.0, -16972.06932406019, 4.333570009765352, 0.0, 0.0, -14707.42362132396, -3.9359863782863593, -126310.12023278416, 0.0, 0.0, 0.0, 2368.7338648797499, -0.79901363129630532, 0.0, 0.0, 72128.060366414618, 1.9360586501514283, 0.0, 0.0, 6067.5004453349093, 3.2592178193264738, 0.0, 0.0, -13907.351529879181, 0.72938468470238693, 0.0, 0.0, -7816.9761497748268, 2.6322124359613301, 64909.319748412963, 0.0, 0.0, 0.0, -23791.780691604748, 1.8910793637963941, 0.0, 0.0, -39372.579704928015, 8.6889053485660135, 0.0, 0.0, 41595.230960589914, -4.3902339481535977, 0.0, 0.0, 10112.272928799337, 0.16506107444965337, 0.0, 0.0, 25731.219155667335, 0.46130897945539479, 0.0, 0.0, 8077.6340416862104, -2.2666842985329221, 0.0, 0.0, 5051.6471594700961, 2.5599044818704457, -18922.572614424953, 0.0, 0.0, 0.0, -27390.862431652142, -5.4254510820269397, 0.0, 0.0, -10626.922204020999, 7.903933319447022, 0.0, 0.0, -7559.5767299673462, 3.9958477189129846, 0.0, 0.0, -5027.7625354094725, 1.4539966351419851, 0.0, 0.0, -12527.616585359714, -2.1579613556381956, 0.0, 0.0, -37550.570759319402, -4.3258197514406094, 0.0, 0.0, -7207.8131427431153, 5.5923007167559113, 0.0, 0.0, -10934.659788070483, 1.5462056815557303, 0.0, 0.0, -16366.37241348215, -1.5385747155059879, 25047.6062182438, 0.0, 0.0, 0.0, 9508.0248644387575, 2.0198784164358696, 0.0, 0.0, -7368.9903795998471, 5.8623209779020256, 0.0, 0.0, 8311.3736131828646, 1.4992901950620758, 0.0, 0.0, 36177.02133195733, 3.3258195341825103, 0.0, 0.0, 11892.087203873632, 5.562433896593574, 0.0, 0.0, 31851.760738854839, -2.1167769777232954, 0.0, 0.0, 30876.553443434816, -5.3089330153539027, 0.0, 0.0, -6285.2177188078704, 4.0232333154977864, 0.0, 0.0, 10832.588384515837, 4.5638689140627484, 0.0, 0.0, 14114.002386595224, 4.2710132435130523, 0.0, 0.0, -4366.4936621754323, -4.2778156918589874, -43204.449024641232, 0.0, 0.0, 0.0, 14511.687075286822, -6.1034572760433603, 0.0, 0.0, 7644.0705070270978, 0.72112747834405599, 0.0, 0.0, -42871.395371724895, 8.5324010467773128, 0.0, 0.0, -22864.270680876944, 12.756935623709822, 0.0, 0.0, 1910.6385733283935, 1.4860435822542193, 0.0, 0.0, -8923.3811275360677, 0.021156295435269104, 0.0, 0.0, -18674.466081287377, -2.5596759713678492, 0.0, 0.0, 21.296692666388893, 2.2396721751367372, 0.0, 0.0, 1605.0199334129031, 1.942000053562025, 0.0, 0.0, -7355.8052917157866, -0.085178977736855899, 0.0, 0.0, -178.73958595323569, 0.17989586109817998, 0.0, 0.0, 4182.3341200215764, 4.3294838255467267, 0.0, 0.0, -5126.0780933967171, 3.3816336872697645, -7175.178402542273, 0.0, 0.0, 0.0, -7308.3848104788049, 1.9755604776273612, 0.0, 0.0, 907.5844769826324, 2.2609766986638453, 0.0, 0.0, 28054.862415791838, -6.2191983569260829, 0.0, 0.0, 12026.118122357248, 17.443247158460427, 0.0, 0.0, 11766.474070554848, 10.494782611360524, 0.0, 0.0, -2275.6047141099889, 3.5015128439030572, 0.0, 0.0, 4743.4119902921057, -3.990800723023967, 0.0, 0.0, 3907.2171582824712, 1.4127861605729506, 0.0, 0.0, -12721.282992510121, 2.2920569832139255, 0.0, 0.0, 649.69038288480999, 1.1212365594407105, 0.0, 0.0, -6102.494402847844, -1.6577620524144914, 0.0, 0.0, -8096.9210297207674, 7.7205045824876493, 0.0, 0.0, -1368.9920705052045, -1.5642694932314585, 0.0, 0.0, 5395.9666599335242, -2.1117569522013517, 0.0, 0.0, -1140.01405051295, 1.9906879133936171, 624.80869328162123, 0.0, 0.0, 0.0, 4089.5349396056608, -3.6121520177209665, 0.0, 0.0, -29968.531010741684, 5.7439378149575493, 0.0, 0.0, -13842.366747404836, -0.083076507030891034, 0.0, 0.0, 9251.74658998427, 10.474210268762711, 0.0, 0.0, -4025.1724808612594, 5.0376737683049297, 0.0, 0.0, 4775.3783570389587, 10.664547039447074, 0.0, 0.0, 11976.191214257957, 0.50201620991965934, 0.0, 0.0, 5867.2639428163502, 1.1896716103905389, 0.0, 0.0, 12877.657067925986, 1.8465660318979775, 0.0, 0.0, 16206.111609433263, 4.8246843405078863, 0.0, 0.0, 3610.4854996584972, 3.3516070780476195, 0.0, 0.0, 9661.5351796437808, -5.9380430259292645, 0.0, 0.0, 8685.2531087868465, -0.64821232890848268, 0.0, 0.0, -312.91711936827596, 9.2168609777382926, 0.0, 0.0, 6298.759850966494, 4.4602580767723747, 0.0, 0.0, 3348.3045543307935, -0.38462078919272813, 0.0, 0.0, 488.51289926704931, -2.1128912645941416, -975.52305859373428, 0.0, 0.0, 0.0, -10755.129903571918, 0.30884068431632339, 0.0, 0.0, 3104.2277033433138, 4.7372665647730328, 0.0, 0.0, 9752.9130898821077, -2.0804460962139686, 0.0, 0.0, -9694.8281227787666, 1.64808120580346, 0.0, 0.0, -1622.3829558462212, 8.5067132512126111, 0.0, 0.0, -11558.805406892518, 14.350648555418546, 0.0, 0.0, 8405.0851462397623, -0.72112972624099048, 0.0, 0.0, 15625.638555235859, -1.3670451022237979, 0.0, 0.0, -3147.7771182901865, 4.8662303313693354, 0.0, 0.0, -4682.4501917588223, 3.6582047312104793, 0.0, 0.0, -13101.93913472442, -1.8255379688292537, 0.0, 0.0, -7556.06062439787, -3.2266823357474479, 0.0, 0.0, -6410.2917754375831, -1.3019240449467371, 0.0, 0.0, -7702.2742811075941, 4.2106258495163953, 0.0, 0.0, 2177.1234415980966, -0.84924429602420592, 0.0, 0.0, -7665.7128282976337, 5.3720471902173932, 0.0, 0.0, 4046.9940657058983, 4.6261175146435791, 0.0, 0.0, -211.63279451514836, 2.4440216742659922, 0.0, 0.0, -1188.9555539240885, -5.6516925299338743, 19695.857081475006, 0.0, 0.0, 0.0, 11252.737508494711, -2.458785211828749, 0.0, 0.0, -7232.0338690955987, 0.27253953774259326, 0.0, 0.0, 1123.7638154174197, 4.8837214488794061, 0.0, 0.0, 1215.3034770015856, 2.1029499497563773, 0.0, 0.0, 2552.3298189366842, -4.7821392334415531, 0.0, 0.0, -529.94244708716974, 22.599584036997086, 0.0, 0.0, 560.40615094398663, 13.309094105051802, 0.0, 0.0, 10242.089928115463, 0.92276408985565261, 0.0, 0.0, -7896.9507053769803, -2.1196969544998265, 0.0, 0.0, 397.11700299896523, -0.49174517770122583]</td>
      </tr>
      <tr>
        <td id="L238" class="blob-num js-line-number" data-line-number="238"></td>
        <td id="LC238" class="blob-code blob-code-inner js-file-line">        ),</td>
      </tr>
      <tr>
        <td id="L239" class="blob-num js-line-number" data-line-number="239"></td>
        <td id="LC239" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;</span>alm0<span class="pl-pds">&#39;</span></span>: n.array(</td>
      </tr>
      <tr>
        <td id="L240" class="blob-num js-line-number" data-line-number="240"></td>
        <td id="LC240" class="blob-code blob-code-inner js-file-line">[-2509.795308092424, 0.0, 0.0, 0.0, -542.95304808539447, -0.089103865486498762, -2674.8921085852267, 0.0, 0.0, 0.0, -2450.8275459460328, -0.12122968266255646, 0.0, 0.0, 484.43258693118355, -0.12444162654967923, 0.0, 0.0, 419.14889644597554, 0.11199842018396657, 3617.1180506112255, 0.0, 0.0, 0.0, -79.345837595100249, 0.024609457915114472, 0.0, 0.0, -2076.1775181825979, -0.055599716031520682, 0.0, 0.0, -171.89694983456775, -0.09107814601572882, 0.0, 0.0, 397.16303975126743, -0.020743466595031901, 0.0, 0.0, 222.00188556890657, -0.074614425553248653, -1848.0514375785062, 0.0, 0.0, 0.0, 677.25212936706521, -0.052427769472071402, 0.0, 0.0, 1122.7561396675696, -0.25026343574461934, 0.0, 0.0, -1177.2398993868574, 0.12448795077559982, 0.0, 0.0, -286.89042984494125, -0.0043671055576578813, 0.0, 0.0, -729.16457065597024, -0.015059272877281931, 0.0, 0.0, -227.54986446468334, 0.064518156065887025, 0.0, 0.0, -144.09526163269231, -0.071896777739056492, 550.56142446132162, 0.0, 0.0, 0.0, 789.95305777447186, 0.15278481548345677, 0.0, 0.0, 302.80826664285104, -0.2292455172036138, 0.0, 0.0, 215.07203955470834, -0.11148641690072135, 0.0, 0.0, 147.6300094316332, -0.041438137068600535, 0.0, 0.0, 356.40663808704829, 0.06020981998761786, 0.0, 0.0, 1068.2809150343332, 0.12059322351429408, 0.0, 0.0, 207.15070625717203, -0.15887907500986867, 0.0, 0.0, 311.51955894051775, -0.044013176018082217, 0.0, 0.0, 465.50514676183599, 0.043086930173945374, -699.91994003413959, 0.0, 0.0, 0.0, -272.30205269131108, -0.058237464090315841, 0.0, 0.0, 206.6696629058103, -0.17113397512837467, 0.0, 0.0, -242.17697804400967, -0.03904696709972983, 0.0, 0.0, -1034.0461250086307, -0.096263366639788267, 0.0, 0.0, -340.36959713926638, -0.157959637524269, 0.0, 0.0, -906.76535925870894, 0.05822882111324016, 0.0, 0.0, -878.19034673650174, 0.15090747340518335, 0.0, 0.0, 178.72710693016728, -0.1133137878039443, 0.0, 0.0, -307.2732605638983, -0.12946761060311943, 0.0, 0.0, -402.15039831465805, -0.12267742876829732, 0.0, 0.0, 124.58755428476331, 0.12160615484632728, 1224.8966172274097, 0.0, 0.0, 0.0, -422.81125499888446, 0.17489349085562872, 0.0, 0.0, -206.13215938425151, -0.024918238081632359, 0.0, 0.0, 1216.0021828633653, -0.23903462262346217, 0.0, 0.0, 646.76999409163841, -0.36686036191363569, 0.0, 0.0, -53.748641101219171, -0.040657195610747071, 0.0, 0.0, 257.01496034832354, -0.00088397200625506657, 0.0, 0.0, 532.92627524275213, 0.073231868546206474, 0.0, 0.0, -2.3123718843121193, -0.063116998506782929, 0.0, 0.0, -45.312170650102566, -0.052918508044701933, 0.0, 0.0, 208.81448662099027, 0.00049665368894422079, 0.0, 0.0, 3.3145191420192557, -0.0075522336446495145, 0.0, 0.0, -118.38406740379477, -0.1235450612088997, 0.0, 0.0, 146.27931891595969, -0.095112431509793954, 216.66543016969877, 0.0, 0.0, 0.0, 202.62032812415652, -0.055071980987297146, 0.0, 0.0, -24.732380939749305, -0.069324586041150008, 0.0, 0.0, -799.12852251187439, 0.17805739075476529, 0.0, 0.0, -338.13613885263197, -0.50258751799301171, 0.0, 0.0, -330.92377568750896, -0.29439745258184846, 0.0, 0.0, 69.955703609112916, -0.099300574356904758, 0.0, 0.0, -135.50283401710723, 0.11336055781808395, 0.0, 0.0, -109.4473303776138, -0.039179921416761625, 0.0, 0.0, 361.89296520495282, -0.063457640307326188, 0.0, 0.0, -19.011530990712416, -0.030296155915062021, 0.0, 0.0, 175.81242107703684, 0.045380812776882037, 0.0, 0.0, 230.1356975837636, -0.21991370682307745, 0.0, 0.0, 38.548063220604263, 0.04447052463281953, 0.0, 0.0, -152.74099867458406, 0.060411014726151104, 0.0, 0.0, 32.799893279193917, -0.05625755938421767, -11.671167858413597, 0.0, 0.0, 0.0, -122.41338733578932, 0.10265761851708999, 0.0, 0.0, 854.8147788993241, -0.16832987784539305, 0.0, 0.0, 389.59483422670991, 0.0010515097421338222, 0.0, 0.0, -263.54000762919696, -0.30602620404886061, 0.0, 0.0, 113.5268590131072, -0.13743520219127883, 0.0, 0.0, -136.1177227410563, -0.30517270252912371, 0.0, 0.0, -347.71640981466663, -0.013328197668937738, 0.0, 0.0, -168.59066374139138, -0.034021615854743656, 0.0, 0.0, -365.89920330695344, -0.05129511900796007, 0.0, 0.0, -459.7707625930787, -0.13649377467013135, 0.0, 0.0, -103.53308934238288, -0.096015678120849338, 0.0, 0.0, -274.34912938228803, 0.16645409598440561, 0.0, 0.0, -247.15567235290939, 0.01807845821834346, 0.0, 0.0, 9.0053572040052821, -0.26370903611326263, 0.0, 0.0, -179.18044921665742, -0.12568774597726795, 0.0, 0.0, -94.892883913068388, 0.010492867772354359, 0.0, 0.0, -13.737508814272434, 0.060075070519419876, 23.415415276762271, 0.0, 0.0, 0.0, 305.72805905313203, -0.0086614331104518417, 0.0, 0.0, -86.195790327242264, -0.13777706251196414, 0.0, 0.0, -274.69250181433779, 0.058458297086366275, 0.0, 0.0, 278.482105784417, -0.053177711988373275, 0.0, 0.0, 44.501303461749991, -0.23822638212032041, 0.0, 0.0, 324.63764111477315, -0.41352873710279964, 0.0, 0.0, -249.4983121147516, 0.021301518932652769, 0.0, 0.0, -454.42693460024361, 0.037898208796734653, 0.0, 0.0, 91.623643436443217, -0.1383467576541032, 0.0, 0.0, 131.72531033239599, -0.10421407678062951, 0.0, 0.0, 372.83411450176357, 0.052076206062910413, 0.0, 0.0, 215.03659811564501, 0.091125817672451065, 0.0, 0.0, 181.72640460655063, 0.036584421520065045, 0.0, 0.0, 219.35570781570911, -0.11636865323780578, 0.0, 0.0, -62.255991452941828, 0.020311442840686849, 0.0, 0.0, 219.08103501446959, -0.15524342747668779, 0.0, 0.0, -114.87121517016, -0.13136769824434186, 0.0, 0.0, 5.9304785270308358, -0.070526493957355579, 0.0, 0.0, 33.931301137524464, 0.16072691130852704, -561.3125186753964, 0.0, 0.0, 0.0, -316.97903860259953, 0.070590200246086601, 0.0, 0.0, 202.91117550362088, -0.009538018080419413, 0.0, 0.0, -31.44408160859355, -0.13909745805711646, 0.0, 0.0, -32.437126472037825, -0.066025505043034366, 0.0, 0.0, -72.232731834313483, 0.13754930097467863, 0.0, 0.0, 16.089915969260041, -0.65224932808333158, 0.0, 0.0, -18.960925699756704, -0.37344958578008358, 0.0, 0.0, -298.63181385705462, -0.027681501480277509, 0.0, 0.0, 229.16214842560413, 0.06063401187656968, 0.0, 0.0, -10.520311488826575, 0.013786299852551351]</td>
      </tr>
      <tr>
        <td id="L241" class="blob-num js-line-number" data-line-number="241"></td>
        <td id="LC241" class="blob-code blob-code-inner js-file-line">        ),</td>
      </tr>
      <tr>
        <td id="L242" class="blob-num js-line-number" data-line-number="242"></td>
        <td id="LC242" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L243" class="blob-num js-line-number" data-line-number="243"></td>
        <td id="LC243" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L244" class="blob-num js-line-number" data-line-number="244"></td>
        <td id="LC244" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L245" class="blob-num js-line-number" data-line-number="245"></td>
        <td id="LC245" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L246" class="blob-num js-line-number" data-line-number="246"></td>
        <td id="LC246" class="blob-code blob-code-inner js-file-line"><span class="pl-k">def</span> <span class="pl-en">get_aa</span>(<span class="pl-smi">freqs</span>):</td>
      </tr>
      <tr>
        <td id="L247" class="blob-num js-line-number" data-line-number="247"></td>
        <td id="LC247" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span>Return the AntennaArray to be used for simulation.<span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L248" class="blob-num js-line-number" data-line-number="248"></td>
        <td id="LC248" class="blob-code blob-code-inner js-file-line">    location <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>loc<span class="pl-pds">&#39;</span></span>]</td>
      </tr>
      <tr>
        <td id="L249" class="blob-num js-line-number" data-line-number="249"></td>
        <td id="LC249" class="blob-code blob-code-inner js-file-line">    antennas <span class="pl-k">=</span> []</td>
      </tr>
      <tr>
        <td id="L250" class="blob-num js-line-number" data-line-number="250"></td>
        <td id="LC250" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> <span class="pl-k">not</span> <span class="pl-s"><span class="pl-pds">&#39;</span>antpos<span class="pl-pds">&#39;</span></span> <span class="pl-k">in</span> prms:</td>
      </tr>
      <tr>
        <td id="L251" class="blob-num js-line-number" data-line-number="251"></td>
        <td id="LC251" class="blob-code blob-code-inner js-file-line">        prms[<span class="pl-s"><span class="pl-pds">&#39;</span>antpos<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>antpos_ideal<span class="pl-pds">&#39;</span></span>]</td>
      </tr>
      <tr>
        <td id="L252" class="blob-num js-line-number" data-line-number="252"></td>
        <td id="LC252" class="blob-code blob-code-inner js-file-line">    nants <span class="pl-k">=</span> <span class="pl-c1">len</span>(prms[<span class="pl-s"><span class="pl-pds">&#39;</span>antpos<span class="pl-pds">&#39;</span></span>])</td>
      </tr>
      <tr>
        <td id="L253" class="blob-num js-line-number" data-line-number="253"></td>
        <td id="LC253" class="blob-code blob-code-inner js-file-line">    antpos_ideal <span class="pl-k">=</span> n.zeros(<span class="pl-v">shape</span><span class="pl-k">=</span>(nants,<span class="pl-c1">3</span>),<span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">float</span>)</td>
      </tr>
      <tr>
        <td id="L254" class="blob-num js-line-number" data-line-number="254"></td>
        <td id="LC254" class="blob-code blob-code-inner js-file-line">    tops <span class="pl-k">=</span> {<span class="pl-s"><span class="pl-pds">&#39;</span>top_x<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">0</span>, <span class="pl-s"><span class="pl-pds">&#39;</span>top_y<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">1</span>, <span class="pl-s"><span class="pl-pds">&#39;</span>top_z<span class="pl-pds">&#39;</span></span>:<span class="pl-c1">2</span>}</td>
      </tr>
      <tr>
        <td id="L255" class="blob-num js-line-number" data-line-number="255"></td>
        <td id="LC255" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">for</span> k <span class="pl-k">in</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>antpos_ideal<span class="pl-pds">&#39;</span></span>].keys():</td>
      </tr>
      <tr>
        <td id="L256" class="blob-num js-line-number" data-line-number="256"></td>
        <td id="LC256" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> i,m <span class="pl-k">in</span> <span class="pl-c1">enumerate</span>(prms[<span class="pl-s"><span class="pl-pds">&#39;</span>antpos_ideal<span class="pl-pds">&#39;</span></span>][k]):</td>
      </tr>
      <tr>
        <td id="L257" class="blob-num js-line-number" data-line-number="257"></td>
        <td id="LC257" class="blob-code blob-code-inner js-file-line">            antpos_ideal[k,tops[m]] <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>antpos_ideal<span class="pl-pds">&#39;</span></span>][k][m]</td>
      </tr>
      <tr>
        <td id="L258" class="blob-num js-line-number" data-line-number="258"></td>
        <td id="LC258" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">for</span> i <span class="pl-k">in</span> <span class="pl-c1">range</span>(nants):</td>
      </tr>
      <tr>
        <td id="L259" class="blob-num js-line-number" data-line-number="259"></td>
        <td id="LC259" class="blob-code blob-code-inner js-file-line">        beam <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>beam<span class="pl-pds">&#39;</span></span>](freqs, nside=<span class="pl-c1">32</span>, lmax=<span class="pl-c1">20</span>, mmax=<span class="pl-c1">20</span>, deg=<span class="pl-c1">7</span>)</td>
      </tr>
      <tr>
        <td id="L260" class="blob-num js-line-number" data-line-number="260"></td>
        <td id="LC260" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">try</span>: beam.set_params(prms[<span class="pl-s"><span class="pl-pds">&#39;</span>bm_prms<span class="pl-pds">&#39;</span></span>])</td>
      </tr>
      <tr>
        <td id="L261" class="blob-num js-line-number" data-line-number="261"></td>
        <td id="LC261" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">except</span>(<span class="pl-c1">AttributeError</span>): <span class="pl-k">pass</span></td>
      </tr>
      <tr>
        <td id="L262" class="blob-num js-line-number" data-line-number="262"></td>
        <td id="LC262" class="blob-code blob-code-inner js-file-line">        phsoff <span class="pl-k">=</span> {<span class="pl-s"><span class="pl-pds">&#39;</span>x<span class="pl-pds">&#39;</span></span>:[<span class="pl-c1">0</span>.,<span class="pl-c1">0</span>.], <span class="pl-s"><span class="pl-pds">&#39;</span>y<span class="pl-pds">&#39;</span></span>:[<span class="pl-c1">0</span>.,<span class="pl-c1">0</span>.]}</td>
      </tr>
      <tr>
        <td id="L263" class="blob-num js-line-number" data-line-number="263"></td>
        <td id="LC263" class="blob-code blob-code-inner js-file-line">        amp <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>amps<span class="pl-pds">&#39;</span></span>].get(i, <span class="pl-c1">4e-3</span>); amp <span class="pl-k">=</span> {<span class="pl-s"><span class="pl-pds">&#39;</span>x<span class="pl-pds">&#39;</span></span>:amp,<span class="pl-s"><span class="pl-pds">&#39;</span>y<span class="pl-pds">&#39;</span></span>:amp}</td>
      </tr>
      <tr>
        <td id="L264" class="blob-num js-line-number" data-line-number="264"></td>
        <td id="LC264" class="blob-code blob-code-inner js-file-line">        bp_r <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>bp_r<span class="pl-pds">&#39;</span></span>][i]; bp_r <span class="pl-k">=</span> {<span class="pl-s"><span class="pl-pds">&#39;</span>x<span class="pl-pds">&#39;</span></span>:bp_r, <span class="pl-s"><span class="pl-pds">&#39;</span>y<span class="pl-pds">&#39;</span></span>:bp_r}</td>
      </tr>
      <tr>
        <td id="L265" class="blob-num js-line-number" data-line-number="265"></td>
        <td id="LC265" class="blob-code blob-code-inner js-file-line">        bp_i <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>bp_i<span class="pl-pds">&#39;</span></span>][i]; bp_i <span class="pl-k">=</span> {<span class="pl-s"><span class="pl-pds">&#39;</span>x<span class="pl-pds">&#39;</span></span>:bp_i, <span class="pl-s"><span class="pl-pds">&#39;</span>y<span class="pl-pds">&#39;</span></span>:bp_i}</td>
      </tr>
      <tr>
        <td id="L266" class="blob-num js-line-number" data-line-number="266"></td>
        <td id="LC266" class="blob-code blob-code-inner js-file-line">        twist <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>twist<span class="pl-pds">&#39;</span></span>][i]</td>
      </tr>
      <tr>
        <td id="L267" class="blob-num js-line-number" data-line-number="267"></td>
        <td id="LC267" class="blob-code blob-code-inner js-file-line">        antennas.append(a.pol.Antenna(<span class="pl-c1">0</span>., <span class="pl-c1">0</span>., <span class="pl-c1">0</span>., beam, <span class="pl-v">phsoff</span><span class="pl-k">=</span>phsoff,</td>
      </tr>
      <tr>
        <td id="L268" class="blob-num js-line-number" data-line-number="268"></td>
        <td id="LC268" class="blob-code blob-code-inner js-file-line">                <span class="pl-v">amp</span><span class="pl-k">=</span>amp, <span class="pl-v">bp_r</span><span class="pl-k">=</span>bp_r, <span class="pl-v">bp_i</span><span class="pl-k">=</span>bp_i, <span class="pl-v">pointing</span><span class="pl-k">=</span>(<span class="pl-c1">0</span>.,n.pi<span class="pl-k">/</span><span class="pl-c1">2</span>,twist)))</td>
      </tr>
      <tr>
        <td id="L269" class="blob-num js-line-number" data-line-number="269"></td>
        <td id="LC269" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>        antennas.append(a.pol.Antenna(0., 0., 0., beam, phsoff=phsoff))</span></td>
      </tr>
      <tr>
        <td id="L270" class="blob-num js-line-number" data-line-number="270"></td>
        <td id="LC270" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>    aa = AntennaArray(prms[&#39;loc&#39;], antennas, tau_ew=prms[&#39;tau_ew&#39;], tau_ns=prms[&#39;tau_ns&#39;],</span></td>
      </tr>
      <tr>
        <td id="L271" class="blob-num js-line-number" data-line-number="271"></td>
        <td id="LC271" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>        gain=prms[&#39;gain&#39;], amp_coeffs=prms[&#39;amp_coeffs&#39;],</span></td>
      </tr>
      <tr>
        <td id="L272" class="blob-num js-line-number" data-line-number="272"></td>
        <td id="LC272" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>        dly_coeffs=prms[&#39;dly_coeffs&#39;], dly_xx_to_yy=prms[&#39;dly_xx_to_yy&#39;], ant_layout=prms[&#39;ant_layout&#39;])</span></td>
      </tr>
      <tr>
        <td id="L273" class="blob-num js-line-number" data-line-number="273"></td>
        <td id="LC273" class="blob-code blob-code-inner js-file-line">    aa <span class="pl-k">=</span> AntennaArray(prms[<span class="pl-s"><span class="pl-pds">&#39;</span>loc<span class="pl-pds">&#39;</span></span>], antennas, <span class="pl-v">antpos_ideal</span><span class="pl-k">=</span>antpos_ideal)</td>
      </tr>
      <tr>
        <td id="L274" class="blob-num js-line-number" data-line-number="274"></td>
        <td id="LC274" class="blob-code blob-code-inner js-file-line">    pos_prms <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L275" class="blob-num js-line-number" data-line-number="275"></td>
        <td id="LC275" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">for</span> i <span class="pl-k">in</span> <span class="pl-c1">range</span>(nants):</td>
      </tr>
      <tr>
        <td id="L276" class="blob-num js-line-number" data-line-number="276"></td>
        <td id="LC276" class="blob-code blob-code-inner js-file-line">        pos_prms[<span class="pl-c1">str</span>(i)] <span class="pl-k">=</span> prms[<span class="pl-s"><span class="pl-pds">&#39;</span>antpos<span class="pl-pds">&#39;</span></span>][i]</td>
      </tr>
      <tr>
        <td id="L277" class="blob-num js-line-number" data-line-number="277"></td>
        <td id="LC277" class="blob-code blob-code-inner js-file-line">    aa.set_params(pos_prms)</td>
      </tr>
      <tr>
        <td id="L278" class="blob-num js-line-number" data-line-number="278"></td>
        <td id="LC278" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> aa</td>
      </tr>
      <tr>
        <td id="L279" class="blob-num js-line-number" data-line-number="279"></td>
        <td id="LC279" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
</table>

  </div>

</div>

<button type="button" data-facebox="#jump-to-line" data-facebox-class="linejump" data-hotkey="l" class="d-none">Jump to Line</button>
<div id="jump-to-line" style="display:none">
  <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="" class="js-jump-to-line-form" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
    <input class="form-control linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
    <button type="submit" class="btn">Go</button>
</form></div>


  </div>
  <div class="modal-backdrop js-touch-events"></div>
</div>

    </div>
  </div>

  </div>

      
<div class="container site-footer-container">
  <div class="site-footer " role="contentinfo">
    <ul class="site-footer-links float-right">
        <li><a href="https://github.com/contact" data-ga-click="Footer, go to contact, text:contact">Contact GitHub</a></li>
      <li><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
      <li><a href="https://shop.github.com" data-ga-click="Footer, go to shop, text:shop">Shop</a></li>
        <li><a href="https://github.com/blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a href="https://github.com/about" data-ga-click="Footer, go to about, text:about">About</a></li>

    </ul>

    <a href="https://github.com" aria-label="Homepage" class="site-footer-mark" title="GitHub">
      <svg aria-hidden="true" class="octicon octicon-mark-github" height="24" version="1.1" viewBox="0 0 16 16" width="24"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>
    <ul class="site-footer-links">
      <li>&copy; 2017 <span title="0.26583s from github-fe123-cp1-prd.iad.github.net">GitHub</span>, Inc.</li>
        <li><a href="https://github.com/site/terms" data-ga-click="Footer, go to terms, text:terms">Terms</a></li>
        <li><a href="https://github.com/site/privacy" data-ga-click="Footer, go to privacy, text:privacy">Privacy</a></li>
        <li><a href="https://github.com/security" data-ga-click="Footer, go to security, text:security">Security</a></li>
        <li><a href="https://status.github.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a href="https://help.github.com" data-ga-click="Footer, go to help, text:help">Help</a></li>
    </ul>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg aria-hidden="true" class="octicon octicon-alert" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.865 1.52c-.18-.31-.51-.5-.87-.5s-.69.19-.87.5L.275 13.5c-.18.31-.18.69 0 1 .19.31.52.5.87.5h13.7c.36 0 .69-.19.86-.5.17-.31.18-.69.01-1L8.865 1.52zM8.995 13h-2v-2h2v2zm0-3h-2V6h2v4z"/></svg>
    <button type="button" class="flash-close js-flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg aria-hidden="true" class="octicon octicon-x" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
    </button>
    You can't perform that action at this time.
  </div>


    
    <script crossorigin="anonymous" integrity="sha256-YR5yZsGniaV+fJRYGk0cFCrPKjyIlL/sjg73oPdyeIc=" src="https://assets-cdn.github.com/assets/frameworks-611e7266c1a789a57e7c94581a4d1c142acf2a3c8894bfec8e0ef7a0f7727887.js"></script>
    <script async="async" crossorigin="anonymous" integrity="sha256-U6TF7czNO1S+W5oUKgn3k2kC95Pc4h/O1/W6xrzb0Kw=" src="https://assets-cdn.github.com/assets/github-53a4c5edcccd3b54be5b9a142a09f7936902f793dce21fced7f5bac6bcdbd0ac.js"></script>
    
    
    
    
  <div class="js-stale-session-flash stale-session-flash flash flash-warn flash-banner d-none">
    <svg aria-hidden="true" class="octicon octicon-alert" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.865 1.52c-.18-.31-.51-.5-.87-.5s-.69.19-.87.5L.275 13.5c-.18.31-.18.69 0 1 .19.31.52.5.87.5h13.7c.36 0 .69-.19.86-.5.17-.31.18-.69.01-1L8.865 1.52zM8.995 13h-2v-2h2v2zm0-3h-2V6h2v4z"/></svg>
    <span class="signed-in-tab-flash">You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="signed-out-tab-flash">You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <div class="facebox" id="facebox" style="display:none;">
  <div class="facebox-popup">
    <div class="facebox-content" role="dialog" aria-labelledby="facebox-header" aria-describedby="facebox-description">
    </div>
    <button type="button" class="facebox-close js-facebox-close" aria-label="Close modal">
      <svg aria-hidden="true" class="octicon octicon-x" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
    </button>
  </div>
</div>


  </body>
</html>

