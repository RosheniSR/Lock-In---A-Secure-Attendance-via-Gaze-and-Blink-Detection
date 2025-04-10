<!DOCTYPE html>
<!-- saved from url=(0105)https://github.com/Rafael-ZP/Lock-In-%2DA_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py -->
<html lang="en" data-color-mode="auto" data-light-theme="light" data-dark-theme="dark" data-a11y-animated-images="system" data-a11y-link-underlines="true" class="js-focus-visible" data-js-focus-visible="" data-turbo-loaded=""><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><style type="text/css">.turbo-progress-bar {
  position: fixed;
  display: block;
  top: 0;
  left: 0;
  height: 3px;
  background: #0076ff;
  z-index: 2147483647;
  transition:
    width 300ms ease-out,
    opacity 150ms 150ms ease-in;
  transform: translate3d(0, 0, 0);
}
</style>
    
  <link rel="dns-prefetch" href="https://github.githubassets.com/">
  <link rel="dns-prefetch" href="https://avatars.githubusercontent.com/">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com/">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">
  <link rel="preconnect" href="https://github.githubassets.com/" crossorigin="">
  <link rel="preconnect" href="https://avatars.githubusercontent.com/">

  


  <link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/light-74231a1f3bbb.css"><link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/dark-8a995f0bacd4.css"><link data-color-theme="dark_dimmed" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_dimmed-f37fb7684b1f.css"><link data-color-theme="dark_high_contrast" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_high_contrast-9ac301c3ebe5.css"><link data-color-theme="dark_colorblind" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_colorblind-cd826e8636dc.css"><link data-color-theme="light_colorblind" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/light_colorblind-f91b0f603451.css"><link data-color-theme="light_high_contrast" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/light_high_contrast-83beb16e0ecf.css"><link data-color-theme="light_tritanopia" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/light_tritanopia-6e122dab64fc.css"><link data-color-theme="dark_tritanopia" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_tritanopia-18119e682df0.css">

    <link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/primer-primitives-225433424a87.css">
    <link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/primer-cba26849680f.css">
    <link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/global-cbc6bc3d1003.css">
    <link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/github-ea73c9cb5377.css">
  <link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/repository-4fce88777fa8.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/code-0210be90f4d3.css">

  


  <script type="application/json" id="client-env">{"locale":"en","featureFlags":["allow_subscription_halted_error","contentful_lp_flex_features_actions","contentful_lp_flex_features_codespaces","contentful_lp_flex_features_code_review","contentful_lp_flex_features_code_search","contentful_lp_flex_features_discussions","contentful_lp_flex_features_issues","copilot_immersive_issue_preview","copilot_new_references_ui","copilot_chat_custom_instructions","copilot_chat_repo_custom_instructions_preview","copilot_chat_show_model_picker_on_retry","copilot_no_floating_button","copilot_topics_as_references","copilot_read_shared_conversation","copilot_duplicate_thread","dotcom_chat_client_side_skills","experimentation_azure_variant_endpoint","failbot_handle_non_errors","geojson_azure_maps","ghost_pilot_confidence_truncation_25","ghost_pilot_confidence_truncation_40","github_models_gateway_parse_params","github_models_o3_mini_streaming","insert_before_patch","issues_advanced_search_has_filter","issues_react_remove_placeholders","issues_react_blur_item_picker_on_close","issues_advanced_search_nested_ownership_filters","issues_dashboard_no_redirects","marketing_pages_search_explore_provider","primer_react_css_modules_ga","react_data_router_pull_requests","remove_child_patch","sample_network_conn_type","swp_enterprise_contact_form","site_copilot_pro_plus","site_proxima_australia_update","viewscreen_sandbox","issues_react_create_milestone","lifecycle_label_name_updates","copilot_task_oriented_assistive_prompts","issues_react_grouped_diff_on_edit_history","issues_react_feature_preview_is_over","refresh_image_video_src","codespaces_prebuild_region_target_update","copilot_code_review_sign_up_closed"]}</script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/wp-runtime-ed98ac6af01f.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_oddbird_popover-polyfill_dist_popover_js-9da652f58479.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_arianotify-polyfill_ariaNotify-polyfill_js-node_modules_github_mi-3abb8f-46b9f4874d95.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_failbot_failbot_ts-952d624642a1.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/environment-f04cb2a9fc8c.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_primer_behaviors_dist_esm_index_mjs-0dbb79f97f8f.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_selector-observer_dist_index_esm_js-f690fd9ae3d5.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_relative-time-element_dist_index_js-62d275b7ddd9.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_text-expander-element_dist_index_js-78748950cb0c.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_auto-complete-element_dist_index_js-node_modules_github_catalyst_-8e9f78-a90ac05d2469.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_filter-input-element_dist_index_js-node_modules_github_remote-inp-b5f1d7-a1760ffda83d.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_markdown-toolbar-element_dist_index_js-ceef33f593fa.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_file-attachment-element_dist_index_js-node_modules_primer_view-co-c44a69-efa32db3a345.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/github-elements-394f8eb34f19.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/element-registry-c20bd0705df8.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_braintree_browser-detection_dist_browser-detection_js-node_modules_githu-2906d7-2a07a295af40.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_lit-html_lit-html_js-be8cb88f481b.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_mini-throttle_dist_index_js-node_modules_morphdom_dist_morphdom-e-7c534c-a4a1922eb55f.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_turbo_dist_turbo_es2017-esm_js-a03ee12d659a.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_remote-form_dist_index_js-node_modules_delegated-events_dist_inde-893f9f-b6294cf703b7.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_color-convert_index_js-e3180fe3bcb3.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_quote-selection_dist_index_js-node_modules_github_session-resume_-947061-e7a6c4a19f98.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_updatable-content_updatable-content_ts-62f3e9c52ece.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/app_assets_modules_github_behaviors_task-list_ts-app_assets_modules_github_sso_ts-ui_packages-900dde-768abe60b1f8.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/app_assets_modules_github_sticky-scroll-into-view_ts-3e000c5d31a9.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/app_assets_modules_github_behaviors_ajax-error_ts-app_assets_modules_github_behaviors_include-d0d0a6-e7f74ee74d91.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/app_assets_modules_github_behaviors_commenting_edit_ts-app_assets_modules_github_behaviors_ht-83c235-4bcbbbfbe1d4.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/behaviors-4414ad8b510b.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_delegated-events_dist_index_js-node_modules_github_catalyst_lib_index_js-f6223d90c7ba.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/notifications-global-01e85cd1be94.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_mini-throttle_dist_index_js-node_modules_github_catalyst_lib_inde-dbbea9-26cce2010167.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/code-menu-d6d3c94ee97e.js.download"></script>
  
  <script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/primer-react-350730ea92ff.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/react-core-a8203875c6f9.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/react-lib-1622bd1e542f.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/octicons-react-cf2f2ab8dab4.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_emotion_is-prop-valid_dist_emotion-is-prop-valid_esm_js-node_modules_emo-41b1a8-555bc0cf9cab.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_mini-throttle_dist_index_js-node_modules_stacktrace-parser_dist_s-e7dcdd-9a233856b02c.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_oddbird_popover-polyfill_dist_popover-fn_js-55fea94174bf.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_dompurify_dist_purify_es_mjs-dd1d3ea6a436.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_lodash-es__Stack_js-node_modules_lodash-es__Uint8Array_js-node_modules_l-4faaa6-4a736fde5c2f.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_lodash-es__baseIsEqual_js-8929eb9718d5.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_react-intersection-observer_react-intersection-observer_modern_mjs-node_-b27033-ba82cef135e3.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_focus-visible_dist_focus-visible_js-node_modules_fzy_js_index_js-node_mo-08d6cf-a84e5768db3f.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_aria-live_aria-live_ts-ui_packages_history_history_ts-ui_packages_promise-with-re-01dc80-b13b6c1d97b0.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_paths_index_ts-8a20a6d3af54.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_ref-selector_RefSelector_tsx-7496afc3784d.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_commit-attribution_index_ts-ui_packages_commit-checks-status_index_ts-ui_packages-762eaa-bac5b6fc3f70.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_code-view-shared_utilities_web-worker_ts-ui_packages_code-view-shared_worker-jobs-7e435b-79b92680ca8b.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_app-uuid_app-uuid_ts-ui_packages_document-metadata_document-metadata_ts-ui_packag-4d8de9-fb57cda8a9d3.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_code-view-shared_hooks_use-canonical-object_ts-ui_packages_code-view-shared_hooks-c2dbff-278e6449f378.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/app_assets_modules_github_blob-anchor_ts-ui_packages_code-nav_code-nav_ts-ui_packages_filter--8253c1-91468a3354f9.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/react-code-view-cfb9f96d7281.js.download"></script>
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/primer-react.eebd62883c61d2053717.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/react-code-view.91744b0963019bd58290.module.css">

  <script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/notifications-subscriptions-menu-53aa08c61b34.js.download"></script>
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/primer-react.eebd62883c61d2053717.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/notifications-subscriptions-menu.1bcff9205c241e99cff2.module.css">


  <title>Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/App.py at main · Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection · GitHub</title>



  <meta name="route-pattern" content="/:user_id/:repository/blob/*name(/*path)" data-turbo-transient="">
  <meta name="route-controller" content="blob" data-turbo-transient="">
  <meta name="route-action" content="show" data-turbo-transient="">

    
  <meta name="current-catalog-service-hash" content="f3abb0cc802f3d7b95fc8762b94bdcb13bf39634c40c357301c4aa1d67a256fb">


  <meta name="request-id" content="B442:20A864:12D6BC:15CB81:67F7D248" data-pjax-transient="true"><meta name="html-safe-nonce" content="1fa5de93d430d7d259f7548ce1be237eb0596adcc7fd1766936047382efd5799" data-pjax-transient="true"><meta name="visitor-payload" content="eyJyZWZlcnJlciI6Imh0dHBzOi8vZ2l0aHViLmNvbS9SYWZhZWwtWlAvTG9jay1Jbi0tQV9TZWN1cmVfQXR0ZW5kYW5jZV92aWFfR2F6ZV9hbmRfQmxpbmtfRGV0ZWN0aW9uL2Jsb2IvbWFpbi9BcHAucHkiLCJyZXF1ZXN0X2lkIjoiQjQ0MjoyMEE4NjQ6MTJENkJDOjE1Q0I4MTo2N0Y3RDI0OCIsInZpc2l0b3JfaWQiOiI0NDI2MjAxNDI3OTI3ODc3NTQ3IiwicmVnaW9uX2VkZ2UiOiJjZW50cmFsaW5kaWEiLCJyZWdpb25fcmVuZGVyIjoiY2VudHJhbGluZGlhIn0=" data-pjax-transient="true"><meta name="visitor-hmac" content="cee699cff3e2c3899844bfcd71de6093d7f0f7a4abd1b6307711bc4d767574bd" data-pjax-transient="true">


    <meta name="hovercard-subject-tag" content="repository:963822950" data-turbo-transient="">


  <meta name="github-keyboard-shortcuts" content="repository,source-code,file-tree,copilot" data-turbo-transient="true">
  

  <meta name="selected-link" value="repo_source" data-turbo-transient="">
  <link rel="assets" href="https://github.githubassets.com/">

    <meta name="google-site-verification" content="Apib7-x98H0j5cPqHWwSMm6dNU4GmODRoqxLiDzdx9I">

<meta name="octolytics-url" content="https://collector.github.com/github/collect">

  <meta name="analytics-location" content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-turbo-transient="true">

  




    <meta name="user-login" content="">

  

    <meta name="viewport" content="width=device-width">

    

      <meta name="description" content="Contribute to Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection development by creating an account on GitHub.">

      <link rel="search" type="application/opensearchdescription+xml" href="https://github.com/opensearch.xml" title="GitHub">

    <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
    <meta property="fb:app_id" content="1401488693436528">
    <meta name="apple-itunes-app" content="app-id=1477376905, app-argument=https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py">

      <meta name="twitter:image" content="https://opengraph.githubassets.com/d04ed04c28881b3ddd07154ec57fd7adde70b424f7fba4b135855e9285b14a41/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection"><meta name="twitter:site" content="@github"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/App.py at main · Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection"><meta name="twitter:description" content="Contribute to Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection development by creating an account on GitHub.">
  <meta property="og:image" content="https://opengraph.githubassets.com/d04ed04c28881b3ddd07154ec57fd7adde70b424f7fba4b135855e9285b14a41/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection"><meta property="og:image:alt" content="Contribute to Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection development by creating an account on GitHub."><meta property="og:image:width" content="1200"><meta property="og:image:height" content="600"><meta property="og:site_name" content="GitHub"><meta property="og:type" content="object"><meta property="og:title" content="Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/App.py at main · Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection"><meta property="og:url" content="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py"><meta property="og:description" content="Contribute to Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection development by creating an account on GitHub.">
  




      <meta name="hostname" content="github.com">



        <meta name="expected-hostname" content="github.com">


  <meta http-equiv="x-pjax-version" content="a74c8e58e975f1b6ea826fa6cebd6e7742481d47eff8c2fff341958181dc84d1" data-turbo-track="reload">
  <meta http-equiv="x-pjax-csp-version" content="e26f9f0ba624ee85cc7ac057d8faa8618a4f25a85eab052c33d018ac0f6b1a46" data-turbo-track="reload">
  <meta http-equiv="x-pjax-css-version" content="3cc343cb09f6c060d681030301f2f209ed6398fe7fc943dea1b120d06a1494a7" data-turbo-track="reload">
  <meta http-equiv="x-pjax-js-version" content="04b8d5e3e49594c3f5759bbcc2df61edb7e4b140816ea4ceb4a0e3e6185fb359" data-turbo-track="reload">

  <meta name="turbo-cache-control" content="no-preview" data-turbo-transient="">

      <meta name="turbo-cache-control" content="no-cache" data-turbo-transient="">

    <meta data-hydrostats="publish">
  <meta name="go-import" content="github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection git https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection.git">

  <meta name="octolytics-dimension-user_id" content="104310982"><meta name="octolytics-dimension-user_login" content="Rafael-ZP"><meta name="octolytics-dimension-repository_id" content="963822950"><meta name="octolytics-dimension-repository_nwo" content="Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection"><meta name="octolytics-dimension-repository_public" content="true"><meta name="octolytics-dimension-repository_is_fork" content="false"><meta name="octolytics-dimension-repository_network_root_id" content="963822950"><meta name="octolytics-dimension-repository_network_root_nwo" content="Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection">



    

    <meta name="turbo-body-classes" content="logged-out env-production page-responsive">


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <meta name="release" content="55338441f0ab2c9e6f307fcc55c0795ff677f3ed">

  <link rel="mask-icon" href="https://github.githubassets.com/assets/pinned-octocat-093da3e6fa40.svg" color="#000000">
  <link rel="alternate icon" class="js-site-favicon" type="image/png" href="https://github.githubassets.com/favicons/favicon.png">
  <link rel="icon" class="js-site-favicon" type="image/svg+xml" href="https://github.githubassets.com/favicons/favicon.svg" data-base-href="https://github.githubassets.com/favicons/favicon">

<meta name="theme-color" content="#1e2327">
<meta name="color-scheme" content="light dark">

  <meta name="msapplication-TileImage" content="/windows-tile.png">
  <meta name="msapplication-TileColor" content="#ffffff">

  <link rel="manifest" href="https://github.com/manifest.json" crossorigin="use-credentials">

  <style data-styled="active" data-styled-version="5.3.11"></style><style id="ms-consent-banner-main-styles">.w8hcgFksdo30C8w-bygqu{color:#000}.ydkKdaztSS0AeHWIeIHsQ a{color:#0067B8}.erL690_8JwUW-R4bJRcfl{background-color:#EBEBEB;border:none;color:#000}.erL690_8JwUW-R4bJRcfl:enabled:hover{color:#000;background-color:#DBDBDB;box-shadow:0px 4px 10px rgba(0,0,0,0.25);border:none}.erL690_8JwUW-R4bJRcfl:enabled:focus{background-color:#DBDBDB;box-shadow:0px 4px 10px rgba(0,0,0,0.25);border:2px solid #000}.erL690_8JwUW-R4bJRcfl:disabled{opacity:1;color:rgba(0,0,0,0.2);background-color:rgba(0,0,0,0.2);border:none}._1zNQOqxpBFSokeCLGi_hGr{border:none;background-color:#0067B8;color:#fff}._1zNQOqxpBFSokeCLGi_hGr:enabled:hover{color:#fff;background-color:#0067B8;box-shadow:0px 4px 10px rgba(0,0,0,0.25);border:none}._1zNQOqxpBFSokeCLGi_hGr:enabled:focus{background-color:#0067B8;box-shadow:0px 4px 10px rgba(0,0,0,0.25);border:2px solid #000}._1zNQOqxpBFSokeCLGi_hGr:disabled{opacity:1;color:rgba(0,0,0,0.2);background-color:rgba(0,120,215,0.2);border:none}._23tra1HsiiP6cT-Cka-ycB{position:relative;display:flex;z-index:9999;width:100%;background-color:#F2F2F2;justify-content:space-between;text-align:left}div[dir="rtl"]._23tra1HsiiP6cT-Cka-ycB{text-align:right}._1Upc2NjY8AlDn177YoVj0y{margin:0;padding-left:5%;padding-top:8px;padding-bottom:8px}div[dir="rtl"] ._1Upc2NjY8AlDn177YoVj0y{margin:0;padding:8px 5% 8px 0;float:none}._23tra1HsiiP6cT-Cka-ycB svg{fill:none;max-width:none;max-height:none}._1V_hlU-7jdtPiooHMu89BB{display:table-cell;padding:12px;width:24px;height:24px;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:24px;line-height:0}.f6QKJD7fhSbnJLarTL-W-{display:table-cell;vertical-align:middle;padding:0;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:13px;line-height:16px}.f6QKJD7fhSbnJLarTL-W- a{text-decoration:underline}._2j0fmugLb1FgYz6KPuB91w{display:inline-block;margin-left:5%;margin-right:5%;min-width:40%;min-width:calc((150px + 3 * 4px) * 2 + 150px);min-width:-webkit-fit-content;min-width:-moz-fit-content;min-width:fit-content;align-self:center;position:relative}._1XuCi2WhiqeWRUVp3pnFG3{margin:4px;padding:5px;min-width:150px;min-height:36px;vertical-align:top;cursor:pointer;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px;text-align:center}._1XuCi2WhiqeWRUVp3pnFG3:focus{box-sizing:border-box}._1XuCi2WhiqeWRUVp3pnFG3:disabled{cursor:not-allowed}._2bvsb3ubApyZ0UGoQA9O9T{display:block;position:fixed;z-index:10000;top:0;left:0;width:100%;height:100%;background-color:rgba(255,255,255,0.6);overflow:auto;text-align:left}div[dir="rtl"]._2bvsb3ubApyZ0UGoQA9O9T{text-align:right}div[dir="rtl"] ._2bvsb3ubApyZ0UGoQA9O9T{left:auto;right:0}.AFsJE948muYyzCMktdzuk{position:relative;top:8%;margin-bottom:40px;margin-left:auto;margin-right:auto;box-sizing:border-box;width:640px;background-color:#fff;border:1px solid #0067B8}._3kWyBRbW_dgnMiEyx06Fu4{float:right;z-index:1;margin:2px;padding:12px;border:none;cursor:pointer;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:13px;line-height:13px;display:flex;align-items:center;text-align:center;color:#666;background-color:#fff}div[dir="rtl"] ._3kWyBRbW_dgnMiEyx06Fu4{margin:2px;padding:12px;float:left}.uCYvKvHXrhjNgflv1VqdD{position:static;margin-top:36px;margin-left:36px;margin-right:36px}._17pX1m9O_W--iZbDt3Ta5r{margin-top:0;margin-bottom:12px;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:600;font-size:20px;line-height:24px;text-transform:none}._1kBkHQ1V1wu3kl-YcLgUr6{height:446px;overflow:auto}._20_nXDf6uFs9Q6wxRXG-I-{margin-top:0;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px}._20_nXDf6uFs9Q6wxRXG-I- a{text-decoration:underline}dl._2a0NH_GDQEQe5Ynfo7suVH{margin-top:36px;margin-bottom:0;padding:0;list-style:none;text-transform:none}dt._3j_LCPv7fyXv3A8FIXVwZ4{margin-top:20px;float:none;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:600;font-size:18px;line-height:24px;list-style:none}.k-vxTGFbdq1aOZB2HHpjh{margin:0;padding:0;border:none}._2Bucyy75c_ogoU1g-liB5R{margin:0;padding:0;border-bottom:none;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:600;font-size:18px;line-height:24px;text-transform:none}._63gwfzV8dclrsl2cfd90r{display:inline-block;margin-top:0;margin-bottom:13px;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px}._1l8wM_4mRYGz3Iu7l3BZR7{display:block}._2UE03QS02aZGkslegN_F-i{display:inline-block;position:relative;left:5px;margin-bottom:13px;margin-right:34px;padding:3px}div[dir="rtl"] ._2UE03QS02aZGkslegN_F-i{margin:0 0 13px 34px;padding:3px;float:none}div[dir="rtl"] ._2UE03QS02aZGkslegN_F-i{left:auto;right:5px}._23tra1HsiiP6cT-Cka-ycB *::before,._2bvsb3ubApyZ0UGoQA9O9T *::before,._23tra1HsiiP6cT-Cka-ycB *::after,._2bvsb3ubApyZ0UGoQA9O9T *::after{box-sizing:inherit}._1HSFn0HzGo6w4ADApV8-c4{outline:2px solid rgba(0,0,0,0.8)}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2{display:inline-block;position:relative;margin-top:0;margin-left:0;margin-right:0;height:0;width:0;border-radius:0;cursor:pointer;outline:none;box-sizing:border-box;-webkit-appearance:none;-moz-appearance:none;appearance:none}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2+label::before{display:block;position:absolute;top:5px;left:3px;height:19px;width:19px;content:"";border-radius:50%;border:1px solid #000;background-color:#fff}div[dir="rtl"] input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2+label::before{left:auto;right:3px}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:hover::before{border:1px solid #0067B8}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:hover::after{display:block;position:absolute;top:10px;left:8px;height:9px;width:9px;content:"";border-radius:50%;background-color:rgba(0,0,0,0.8)}div[dir="rtl"] input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:hover::after{left:auto;right:8px}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:focus::before{border:1px solid #0067B8}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:focus::after{display:block;position:absolute;top:10px;left:8px;height:9px;width:9px;content:"";border-radius:50%;background-color:#000}div[dir="rtl"] input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:focus::after{left:auto;right:8px}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:checked+label::after{display:block;position:absolute;top:10px;left:8px;height:9px;width:9px;content:"";border-radius:50%;background-color:#000}div[dir="rtl"] input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:checked+label::after{left:auto;right:8px}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:disabled+label{cursor:not-allowed}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:disabled+label::before{border:1px solid rgba(0,0,0,0.2);background-color:rgba(0,0,0,0.2)}._3RJzeL3l9Rl_lAQEm6VwdX{display:block;position:static;float:right;margin-top:0;margin-bottom:0;margin-left:19px;margin-right:0;padding-top:0;padding-bottom:0;padding-left:8px;padding-right:0;width:80%;width:calc(100% - 19px);font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px;text-transform:none;cursor:pointer;box-sizing:border-box}div[dir="rtl"] ._3RJzeL3l9Rl_lAQEm6VwdX{margin:0 19px 0 0;padding:0 8px 0 0;float:left}.nohp3sIG12ZBhzcMnPala{margin-top:20px;margin-bottom:48px}._2uhaEsmeotZ3P-M0AXo2kF{padding:0;width:278px;height:36px;cursor:pointer;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px;text-align:center}._2uhaEsmeotZ3P-M0AXo2kF:focus{box-sizing:border-box}._2uhaEsmeotZ3P-M0AXo2kF:disabled{cursor:not-allowed}._3tOu1FJ59c_xz_PmI1lKV5{float:right;padding:0;width:278px;height:36px;cursor:pointer;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px;text-align:center}._3tOu1FJ59c_xz_PmI1lKV5:focus{box-sizing:border-box}._3tOu1FJ59c_xz_PmI1lKV5:disabled{cursor:not-allowed}div[dir="rtl"] ._3tOu1FJ59c_xz_PmI1lKV5{margin:0;padding:0;float:left}@media only screen and (max-width: 768px){._2j0fmugLb1FgYz6KPuB91w,._1Upc2NjY8AlDn177YoVj0y{padding-top:8px;padding-bottom:12px;padding-left:3.75%;padding-right:3.75%;margin:0;width:92.5%}._23tra1HsiiP6cT-Cka-ycB{display:block}._1XuCi2WhiqeWRUVp3pnFG3{margin-bottom:8px;margin-left:0;margin-right:0;width:100%}._2bvsb3ubApyZ0UGoQA9O9T{overflow:hidden}.AFsJE948muYyzCMktdzuk{top:1.8%;width:93.33%;height:96.4%;overflow:hidden}.uCYvKvHXrhjNgflv1VqdD{margin-top:24px;margin-left:24px;margin-right:24px;height:100%}._1kBkHQ1V1wu3kl-YcLgUr6{height:62%;height:calc(100% - 188px);min-height:50%}._2uhaEsmeotZ3P-M0AXo2kF{width:100%}._3tOu1FJ59c_xz_PmI1lKV5{margin-bottom:12px;margin-left:0;width:100%}div[dir="rtl"] ._3tOu1FJ59c_xz_PmI1lKV5{margin:0 0 12px 0;padding:0;float:none}}@media only screen and (max-width: 768px) and (orientation: landscape), only screen and (max-height: 260px), only screen and (max-width: 340px){.AFsJE948muYyzCMktdzuk{overflow:auto}}@media only screen and (max-height: 260px), only screen and (max-width: 340px){._1XuCi2WhiqeWRUVp3pnFG3{min-width:0}._3kWyBRbW_dgnMiEyx06Fu4{padding:3%}.uCYvKvHXrhjNgflv1VqdD{margin-top:3%;margin-left:3%;margin-right:3%}._17pX1m9O_W--iZbDt3Ta5r{margin-bottom:3%}._1kBkHQ1V1wu3kl-YcLgUr6{height:calc(79% - 64px)}.nohp3sIG12ZBhzcMnPala{margin-top:5%;margin-bottom:10%}._3tOu1FJ59c_xz_PmI1lKV5{margin-bottom:3%}div[dir="rtl"] ._3tOu1FJ59c_xz_PmI1lKV5{margin:0 0 3% 0;padding:0;float:none}}
</style><style type="text/css" id="ms-consent-banner-theme-styles">._23tra1HsiiP6cT-Cka-ycB {
            background-color: #24292f !important;
        }.w8hcgFksdo30C8w-bygqu {
            color: #ffffff !important;
        }.ydkKdaztSS0AeHWIeIHsQ a {
            color: #d8b9ff !important;
        }._2bvsb3ubApyZ0UGoQA9O9T {
            background-color: rgba(23, 23, 23, 0.8) !important;
        }.AFsJE948muYyzCMktdzuk {
            background-color: #24292f !important;
            border: 1px solid #d8b9ff !important;
        }._3kWyBRbW_dgnMiEyx06Fu4 {
            color: #d8b9ff !important;
            background-color: #24292f !important;
        }._1zNQOqxpBFSokeCLGi_hGr {
            border: 1px solid #ffffff !important;
            background-color: #ffffff !important;
            color: #1f2328 !important;
        }._1zNQOqxpBFSokeCLGi_hGr:enabled:hover {
            color: #1f2328 !important;
            background-color: #d8b9ff !important;
            box-shadow: none !important;
            border: 1px solid transparent !important;
        }._1zNQOqxpBFSokeCLGi_hGr:enabled:focus {
            background-color: #d8b9ff !important;
            box-shadow: none !important;
            border: 2px solid #ffffff !important;
        }._1zNQOqxpBFSokeCLGi_hGr:disabled {
            opacity: 0.5 !important;
            color: #1f2328 !important;
            background-color: #ffffff !important;
            border: 1px solid transparent !important;
        }.erL690_8JwUW-R4bJRcfl {
            border: 1px solid #eaeef2 !important;
            background-color: #32383f !important;
            color: #ffffff !important;
        }.erL690_8JwUW-R4bJRcfl:enabled:hover {
            color: #ffffff !important;
            background-color: #24292f !important;
            box-shadow: none !important;
            border: 1px solid #ffffff !important;
        }.erL690_8JwUW-R4bJRcfl:enabled:focus {
            background-color: #24292f !important;
            box-shadow: none !important;
            border: 2px solid #6e7781 !important;
        }.erL690_8JwUW-R4bJRcfl:disabled {
            opacity: 0.5 !important;
            color: #ffffff !important;
            background-color: #424a53 !important;
            border: 1px solid #6e7781 !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label::before {
            border: 1px solid #d8b9ff !important;
            background-color: #24292f !important;
        }._1HSFn0HzGo6w4ADApV8-c4 {
            outline: 2px solid #ffffff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:checked + label::after {
            background-color: #d8b9ff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label:hover::before {
            border: 1px solid #ffffff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label:hover::after {
            background-color: #ffffff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label:focus::before {
            border: 1px solid #ffffff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label:focus::after {
            background-color: #d8b9ff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:disabled + label::before {
            border: 1px solid rgba(227, 227, 227, 0.2) !important;
            background-color: rgba(227, 227, 227, 0.2) !important;
        }</style></head>

  <body class="logged-out env-production page-responsive" style="overflow-wrap: break-word; --dialog-scrollgutter: 15px;">
    <div data-turbo-body="" class="logged-out env-production page-responsive" style="word-wrap: break-word;">
      


    <div class="position-relative header-wrapper js-header-wrapper ">
      <a href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py#start-of-content" data-skip-target-assigned="false" class="px-2 py-4 color-bg-accent-emphasis color-fg-on-emphasis show-on-focus js-skip-to-content">Skip to content</a>

      <span data-view-component="true" class="progress-pjax-loader Progress position-fixed width-full">
    <span style="width: 0%;" data-view-component="true" class="Progress-item progress-pjax-loader-bar left-0 top-0 color-bg-accent-emphasis"></span>
</span>      
      
      <script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/ui_packages_ui-commands_ui-commands_ts-2d52c8e72e64.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/keyboard-shortcuts-dialog-2560f573c7ca.js.download"></script>
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./App_files/primer-react.eebd62883c61d2053717.module.css">

<react-partial partial-name="keyboard-shortcuts-dialog" data-ssr="false" data-attempted-ssr="false" data-catalyst="" class="loaded">
  
  <script type="application/json" data-target="react-partial.embeddedData">{"props":{"docsUrl":"https://docs.github.com/get-started/accessibility/keyboard-shortcuts"}}</script>
  <div data-target="react-partial.reactRoot"><div class="d-none"></div><script type="application/json" id="__PRIMER_DATA_:r1i:__">{"resolvedServerColorMode":"day"}</script></div>
</react-partial>




      

          

              
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/vendors-node_modules_github_remote-form_dist_index_js-node_modules_delegated-events_dist_inde-94fd67-4898d1bf4b51.js.download"></script>
<script crossorigin="anonymous" defer="defer" type="application/javascript" src="./App_files/sessions-45d6658f8b6b.js.download"></script>
<header class="HeaderMktg header-logged-out js-details-container js-header Details f4 py-3" role="banner" data-is-top="true" data-color-mode="light" data-light-theme="light" data-dark-theme="dark">
  <h2 class="sr-only">Navigation Menu</h2>

  <button type="button" class="HeaderMktg-backdrop d-lg-none border-0 position-fixed top-0 left-0 width-full height-full js-details-target" aria-label="Toggle navigation">
    <span class="d-none">Toggle navigation</span>
  </button>

  <div class="d-flex flex-column flex-lg-row flex-items-center px-3 px-md-4 px-lg-5 height-full position-relative z-1">
    <div class="d-flex flex-justify-between flex-items-center width-full width-lg-auto">
      <div class="flex-1">
        <button aria-label="Toggle navigation" aria-expanded="false" type="button" data-view-component="true" class="js-details-target js-nav-padding-recalculate js-header-menu-toggle Button--link Button--medium Button d-lg-none color-fg-inherit p-1">  <span class="Button-content">
    <span class="Button-label"><div class="HeaderMenu-toggle-bar rounded my-1"></div>
            <div class="HeaderMenu-toggle-bar rounded my-1"></div>
            <div class="HeaderMenu-toggle-bar rounded my-1"></div></span>
  </span>
</button>
      </div>

      <a class="mr-lg-3 color-fg-inherit flex-order-2 js-prevent-focus-on-mobile-nav" href="https://github.com/" aria-label="Homepage" data-analytics-event="{&quot;category&quot;:&quot;Marketing nav&quot;,&quot;action&quot;:&quot;click to go to homepage&quot;,&quot;label&quot;:&quot;ref_page:Marketing;ref_cta:Logomark;ref_loc:Header&quot;}">
        <svg height="32" aria-hidden="true" viewBox="0 0 24 24" version="1.1" width="32" data-view-component="true" class="octicon octicon-mark-github">
    <path d="M12 1C5.9225 1 1 5.9225 1 12C1 16.8675 4.14875 20.9787 8.52125 22.4362C9.07125 22.5325 9.2775 22.2025 9.2775 21.9137C9.2775 21.6525 9.26375 20.7862 9.26375 19.865C6.5 20.3737 5.785 19.1912 5.565 18.5725C5.44125 18.2562 4.905 17.28 4.4375 17.0187C4.0525 16.8125 3.5025 16.3037 4.42375 16.29C5.29 16.2762 5.90875 17.0875 6.115 17.4175C7.105 19.0812 8.68625 18.6137 9.31875 18.325C9.415 17.61 9.70375 17.1287 10.02 16.8537C7.5725 16.5787 5.015 15.63 5.015 11.4225C5.015 10.2262 5.44125 9.23625 6.1425 8.46625C6.0325 8.19125 5.6475 7.06375 6.2525 5.55125C6.2525 5.55125 7.17375 5.2625 9.2775 6.67875C10.1575 6.43125 11.0925 6.3075 12.0275 6.3075C12.9625 6.3075 13.8975 6.43125 14.7775 6.67875C16.8813 5.24875 17.8025 5.55125 17.8025 5.55125C18.4075 7.06375 18.0225 8.19125 17.9125 8.46625C18.6138 9.23625 19.04 10.2125 19.04 11.4225C19.04 15.6437 16.4688 16.5787 14.0213 16.8537C14.42 17.1975 14.7638 17.8575 14.7638 18.8887C14.7638 20.36 14.75 21.5425 14.75 21.9137C14.75 22.2025 14.9563 22.5462 15.5063 22.4362C19.8513 20.9787 23 16.8537 23 12C23 5.9225 18.0775 1 12 1Z"></path>
</svg>
      </a>

      <div class="flex-1 flex-order-2 text-right">
          <a href="https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2FRafael-ZP%2FLock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection%2Fblob%2Fmain%2FApp.py" class="HeaderMenu-link HeaderMenu-button d-inline-flex d-lg-none flex-order-1 f5 no-underline border color-border-default rounded-2 px-2 py-1 color-fg-inherit js-prevent-focus-on-mobile-nav" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="e4964736d58e49e309c7e45dff1f30c1bdb8f1d0516061fe681b79be81e56de3" data-analytics-event="{&quot;category&quot;:&quot;Marketing nav&quot;,&quot;action&quot;:&quot;click to Sign in&quot;,&quot;label&quot;:&quot;ref_page:Marketing;ref_cta:Sign in;ref_loc:Header&quot;}">
            Sign in
          </a>
      </div>
    </div>


    <div class="HeaderMenu js-header-menu height-fit position-lg-relative d-lg-flex flex-column flex-auto top-0">
      <div class="HeaderMenu-wrapper d-flex flex-column flex-self-start flex-lg-row flex-auto rounded rounded-lg-0">
          <nav class="HeaderMenu-nav" aria-label="Global">
            <ul class="d-lg-flex list-style-none">
                <li class="HeaderMenu-item position-relative flex-wrap flex-justify-between flex-items-center d-block d-lg-flex flex-lg-nowrap flex-lg-items-center js-details-container js-header-menu-item">
      <button type="button" class="HeaderMenu-link border-0 width-full width-lg-auto px-0 px-lg-2 py-lg-2 no-wrap d-flex flex-items-center flex-justify-between js-details-target" aria-expanded="false">
        Product
        <svg opacity="0.5" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-chevron-down HeaderMenu-icon ml-1">
    <path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path>
</svg>
      </button>
      <div class="HeaderMenu-dropdown dropdown-menu rounded m-0 p-0 pt-2 pt-lg-4 position-relative position-lg-absolute left-0 left-lg-n3 pb-2 pb-lg-4 d-lg-flex flex-wrap dropdown-menu-wide">
          <div class="HeaderMenu-column px-lg-4 border-lg-right mb-4 mb-lg-0 pr-lg-7">
              <div class="border-bottom pb-3 pb-lg-0 border-lg-bottom-0">
                <ul class="list-style-none f5">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description pb-lg-3" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;github_copilot&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;github_copilot_link_product_navbar&quot;}" href="https://github.com/features/copilot">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-copilot color-fg-subtle mr-3">
    <path d="M23.922 16.992c-.861 1.495-5.859 5.023-11.922 5.023-6.063 0-11.061-3.528-11.922-5.023A.641.641 0 0 1 0 16.736v-2.869a.841.841 0 0 1 .053-.22c.372-.935 1.347-2.292 2.605-2.656.167-.429.414-1.055.644-1.517a10.195 10.195 0 0 1-.052-1.086c0-1.331.282-2.499 1.132-3.368.397-.406.89-.717 1.474-.952 1.399-1.136 3.392-2.093 6.122-2.093 2.731 0 4.767.957 6.166 2.093.584.235 1.077.546 1.474.952.85.869 1.132 2.037 1.132 3.368 0 .368-.014.733-.052 1.086.23.462.477 1.088.644 1.517 1.258.364 2.233 1.721 2.605 2.656a.832.832 0 0 1 .053.22v2.869a.641.641 0 0 1-.078.256ZM12.172 11h-.344a4.323 4.323 0 0 1-.355.508C10.703 12.455 9.555 13 7.965 13c-1.725 0-2.989-.359-3.782-1.259a2.005 2.005 0 0 1-.085-.104L4 11.741v6.585c1.435.779 4.514 2.179 8 2.179 3.486 0 6.565-1.4 8-2.179v-6.585l-.098-.104s-.033.045-.085.104c-.793.9-2.057 1.259-3.782 1.259-1.59 0-2.738-.545-3.508-1.492a4.323 4.323 0 0 1-.355-.508h-.016.016Zm.641-2.935c.136 1.057.403 1.913.878 2.497.442.544 1.134.938 2.344.938 1.573 0 2.292-.337 2.657-.751.384-.435.558-1.15.558-2.361 0-1.14-.243-1.847-.705-2.319-.477-.488-1.319-.862-2.824-1.025-1.487-.161-2.192.138-2.533.529-.269.307-.437.808-.438 1.578v.021c0 .265.021.562.063.893Zm-1.626 0c.042-.331.063-.628.063-.894v-.02c-.001-.77-.169-1.271-.438-1.578-.341-.391-1.046-.69-2.533-.529-1.505.163-2.347.537-2.824 1.025-.462.472-.705 1.179-.705 2.319 0 1.211.175 1.926.558 2.361.365.414 1.084.751 2.657.751 1.21 0 1.902-.394 2.344-.938.475-.584.742-1.44.878-2.497Z"></path><path d="M14.5 14.25a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1Zm-5 0a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">GitHub Copilot</div>
        Write better code with AI
      </div>

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description pb-lg-3" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;github_advanced_security&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;github_advanced_security_link_product_navbar&quot;}" href="https://github.com/security/advanced-security">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-shield-check color-fg-subtle mr-3">
    <path d="M16.53 9.78a.75.75 0 0 0-1.06-1.06L11 13.19l-1.97-1.97a.75.75 0 0 0-1.06 1.06l2.5 2.5a.75.75 0 0 0 1.06 0l5-5Z"></path><path d="m12.54.637 8.25 2.675A1.75 1.75 0 0 1 22 4.976V10c0 6.19-3.771 10.704-9.401 12.83a1.704 1.704 0 0 1-1.198 0C5.77 20.705 2 16.19 2 10V4.976c0-.758.489-1.43 1.21-1.664L11.46.637a1.748 1.748 0 0 1 1.08 0Zm-.617 1.426-8.25 2.676a.249.249 0 0 0-.173.237V10c0 5.46 3.28 9.483 8.43 11.426a.199.199 0 0 0 .14 0C17.22 19.483 20.5 15.461 20.5 10V4.976a.25.25 0 0 0-.173-.237l-8.25-2.676a.253.253 0 0 0-.154 0Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">GitHub Advanced Security</div>
        Find and fix vulnerabilities
      </div>

    
</a></li>

                    
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description pb-lg-3" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;actions&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;actions_link_product_navbar&quot;}" href="https://github.com/features/actions">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-workflow color-fg-subtle mr-3">
    <path d="M1 3a2 2 0 0 1 2-2h6.5a2 2 0 0 1 2 2v6.5a2 2 0 0 1-2 2H7v4.063C7 16.355 7.644 17 8.438 17H12.5v-2.5a2 2 0 0 1 2-2H21a2 2 0 0 1 2 2V21a2 2 0 0 1-2 2h-6.5a2 2 0 0 1-2-2v-2.5H8.437A2.939 2.939 0 0 1 5.5 15.562V11.5H3a2 2 0 0 1-2-2Zm2-.5a.5.5 0 0 0-.5.5v6.5a.5.5 0 0 0 .5.5h6.5a.5.5 0 0 0 .5-.5V3a.5.5 0 0 0-.5-.5ZM14.5 14a.5.5 0 0 0-.5.5V21a.5.5 0 0 0 .5.5H21a.5.5 0 0 0 .5-.5v-6.5a.5.5 0 0 0-.5-.5Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">Actions</div>
        Automate any workflow
      </div>

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description pb-lg-3" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;codespaces&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;codespaces_link_product_navbar&quot;}" href="https://github.com/features/codespaces">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-codespaces color-fg-subtle mr-3">
    <path d="M3.5 3.75C3.5 2.784 4.284 2 5.25 2h13.5c.966 0 1.75.784 1.75 1.75v7.5A1.75 1.75 0 0 1 18.75 13H5.25a1.75 1.75 0 0 1-1.75-1.75Zm-2 12c0-.966.784-1.75 1.75-1.75h17.5c.966 0 1.75.784 1.75 1.75v4a1.75 1.75 0 0 1-1.75 1.75H3.25a1.75 1.75 0 0 1-1.75-1.75ZM5.25 3.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h13.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Zm-2 12a.25.25 0 0 0-.25.25v4c0 .138.112.25.25.25h17.5a.25.25 0 0 0 .25-.25v-4a.25.25 0 0 0-.25-.25Z"></path><path d="M10 17.75a.75.75 0 0 1 .75-.75h6.5a.75.75 0 0 1 0 1.5h-6.5a.75.75 0 0 1-.75-.75Zm-4 0a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 0 1.5h-.5a.75.75 0 0 1-.75-.75Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">Codespaces</div>
        Instant dev environments
      </div>

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description pb-lg-3" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;issues&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;issues_link_product_navbar&quot;}" href="https://github.com/features/issues">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-issue-opened color-fg-subtle mr-3">
    <path d="M12 1c6.075 0 11 4.925 11 11s-4.925 11-11 11S1 18.075 1 12 5.925 1 12 1ZM2.5 12a9.5 9.5 0 0 0 9.5 9.5 9.5 9.5 0 0 0 9.5-9.5A9.5 9.5 0 0 0 12 2.5 9.5 9.5 0 0 0 2.5 12Zm9.5 2a2 2 0 1 1-.001-3.999A2 2 0 0 1 12 14Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">Issues</div>
        Plan and track work
      </div>

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description pb-lg-3" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;code_review&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;code_review_link_product_navbar&quot;}" href="https://github.com/features/code-review">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-code-review color-fg-subtle mr-3">
    <path d="M10.3 6.74a.75.75 0 0 1-.04 1.06l-2.908 2.7 2.908 2.7a.75.75 0 1 1-1.02 1.1l-3.5-3.25a.75.75 0 0 1 0-1.1l3.5-3.25a.75.75 0 0 1 1.06.04Zm3.44 1.06a.75.75 0 1 1 1.02-1.1l3.5 3.25a.75.75 0 0 1 0 1.1l-3.5 3.25a.75.75 0 1 1-1.02-1.1l2.908-2.7-2.908-2.7Z"></path><path d="M1.5 4.25c0-.966.784-1.75 1.75-1.75h17.5c.966 0 1.75.784 1.75 1.75v12.5a1.75 1.75 0 0 1-1.75 1.75h-9.69l-3.573 3.573A1.458 1.458 0 0 1 5 21.043V18.5H3.25a1.75 1.75 0 0 1-1.75-1.75ZM3.25 4a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h2.5a.75.75 0 0 1 .75.75v3.19l3.72-3.72a.749.749 0 0 1 .53-.22h10a.25.25 0 0 0 .25-.25V4.25a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">Code Review</div>
        Manage code changes
      </div>

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description pb-lg-3" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;discussions&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;discussions_link_product_navbar&quot;}" href="https://github.com/features/discussions">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-comment-discussion color-fg-subtle mr-3">
    <path d="M1.75 1h12.5c.966 0 1.75.784 1.75 1.75v9.5A1.75 1.75 0 0 1 14.25 14H8.061l-2.574 2.573A1.458 1.458 0 0 1 3 15.543V14H1.75A1.75 1.75 0 0 1 0 12.25v-9.5C0 1.784.784 1 1.75 1ZM1.5 2.75v9.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h6.5a.25.25 0 0 0 .25-.25v-9.5a.25.25 0 0 0-.25-.25H1.75a.25.25 0 0 0-.25.25Z"></path><path d="M22.5 8.75a.25.25 0 0 0-.25-.25h-3.5a.75.75 0 0 1 0-1.5h3.5c.966 0 1.75.784 1.75 1.75v9.5A1.75 1.75 0 0 1 22.25 20H21v1.543a1.457 1.457 0 0 1-2.487 1.03L15.939 20H10.75A1.75 1.75 0 0 1 9 18.25v-1.465a.75.75 0 0 1 1.5 0v1.465c0 .138.112.25.25.25h5.5a.75.75 0 0 1 .53.22l2.72 2.72v-2.19a.75.75 0 0 1 .75-.75h2a.25.25 0 0 0 .25-.25v-9.5Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">Discussions</div>
        Collaborate outside of code
      </div>

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;code_search&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;code_search_link_product_navbar&quot;}" href="https://github.com/features/code-search">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-code-square color-fg-subtle mr-3">
    <path d="M10.3 8.24a.75.75 0 0 1-.04 1.06L7.352 12l2.908 2.7a.75.75 0 1 1-1.02 1.1l-3.5-3.25a.75.75 0 0 1 0-1.1l3.5-3.25a.75.75 0 0 1 1.06.04Zm3.44 1.06a.75.75 0 1 1 1.02-1.1l3.5 3.25a.75.75 0 0 1 0 1.1l-3.5 3.25a.75.75 0 1 1-1.02-1.1l2.908-2.7-2.908-2.7Z"></path><path d="M2 3.75C2 2.784 2.784 2 3.75 2h16.5c.966 0 1.75.784 1.75 1.75v16.5A1.75 1.75 0 0 1 20.25 22H3.75A1.75 1.75 0 0 1 2 20.25Zm1.75-.25a.25.25 0 0 0-.25.25v16.5c0 .138.112.25.25.25h16.5a.25.25 0 0 0 .25-.25V3.75a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">Code Search</div>
        Find more, search less
      </div>

    
</a></li>

                </ul>
              </div>
          </div>
          <div class="HeaderMenu-column px-lg-4">
              <div class="border-bottom pb-3 pb-lg-0 border-lg-bottom-0 border-bottom-0">
                    <span class="d-block h4 color-fg-default my-1" id="product-explore-heading">Explore</span>
                <ul class="list-style-none f5" aria-labelledby="product-explore-heading">
                    
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;all_features&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;all_features_link_product_navbar&quot;}" href="https://github.com/features">
      All features

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary Link--external" target="_blank" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;documentation&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;documentation_link_product_navbar&quot;}" href="https://docs.github.com/">
      Documentation

    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-link-external HeaderMenu-external-icon color-fg-subtle">
    <path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path>
</svg>
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary Link--external" target="_blank" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;github_skills&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;github_skills_link_product_navbar&quot;}" href="https://skills.github.com/">
      GitHub Skills

    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-link-external HeaderMenu-external-icon color-fg-subtle">
    <path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path>
</svg>
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary Link--external" target="_blank" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;blog&quot;,&quot;context&quot;:&quot;product&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;blog_link_product_navbar&quot;}" href="https://github.blog/">
      Blog

    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-link-external HeaderMenu-external-icon color-fg-subtle">
    <path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path>
</svg>
</a></li>

                </ul>
              </div>
          </div>
      </div>
</li>


                <li class="HeaderMenu-item position-relative flex-wrap flex-justify-between flex-items-center d-block d-lg-flex flex-lg-nowrap flex-lg-items-center js-details-container js-header-menu-item">
      <button type="button" class="HeaderMenu-link border-0 width-full width-lg-auto px-0 px-lg-2 py-lg-2 no-wrap d-flex flex-items-center flex-justify-between js-details-target" aria-expanded="false">
        Solutions
        <svg opacity="0.5" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-chevron-down HeaderMenu-icon ml-1">
    <path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path>
</svg>
      </button>
      <div class="HeaderMenu-dropdown dropdown-menu rounded m-0 p-0 pt-2 pt-lg-4 position-relative position-lg-absolute left-0 left-lg-n3 d-lg-flex flex-wrap dropdown-menu-wide">
          <div class="HeaderMenu-column px-lg-4 border-lg-right mb-4 mb-lg-0 pr-lg-7">
              <div class="border-bottom pb-3 pb-lg-0 border-lg-bottom-0 pb-lg-3 mb-3 mb-lg-0">
                    <span class="d-block h4 color-fg-default my-1" id="solutions-by-company-size-heading">By company size</span>
                <ul class="list-style-none f5" aria-labelledby="solutions-by-company-size-heading">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;enterprises&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;enterprises_link_solutions_navbar&quot;}" href="https://github.com/enterprise">
      Enterprises

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;small_and_medium_teams&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;small_and_medium_teams_link_solutions_navbar&quot;}" href="https://github.com/team">
      Small and medium teams

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;startups&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;startups_link_solutions_navbar&quot;}" href="https://github.com/enterprise/startups">
      Startups

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;nonprofits&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;nonprofits_link_solutions_navbar&quot;}" href="https://github.com/solutions/industry/nonprofits">
      Nonprofits

    
</a></li>

                </ul>
              </div>
              <div class="border-bottom pb-3 pb-lg-0 border-lg-bottom-0">
                    <span class="d-block h4 color-fg-default my-1" id="solutions-by-use-case-heading">By use case</span>
                <ul class="list-style-none f5" aria-labelledby="solutions-by-use-case-heading">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;devsecops&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;devsecops_link_solutions_navbar&quot;}" href="https://github.com/solutions/use-case/devsecops">
      DevSecOps

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;devops&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;devops_link_solutions_navbar&quot;}" href="https://github.com/solutions/use-case/devops">
      DevOps

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;ci_cd&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;ci_cd_link_solutions_navbar&quot;}" href="https://github.com/solutions/use-case/ci-cd">
      CI/CD

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;view_all_use_cases&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;view_all_use_cases_link_solutions_navbar&quot;}" href="https://github.com/solutions/use-case">
      View all use cases

    
</a></li>

                </ul>
              </div>
          </div>
          <div class="HeaderMenu-column px-lg-4">
              <div class="border-bottom pb-3 pb-lg-0 border-lg-bottom-0">
                    <span class="d-block h4 color-fg-default my-1" id="solutions-by-industry-heading">By industry</span>
                <ul class="list-style-none f5" aria-labelledby="solutions-by-industry-heading">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;healthcare&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;healthcare_link_solutions_navbar&quot;}" href="https://github.com/solutions/industry/healthcare">
      Healthcare

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;financial_services&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;financial_services_link_solutions_navbar&quot;}" href="https://github.com/solutions/industry/financial-services">
      Financial services

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;manufacturing&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;manufacturing_link_solutions_navbar&quot;}" href="https://github.com/solutions/industry/manufacturing">
      Manufacturing

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;government&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;government_link_solutions_navbar&quot;}" href="https://github.com/solutions/industry/government">
      Government

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;view_all_industries&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;view_all_industries_link_solutions_navbar&quot;}" href="https://github.com/solutions/industry">
      View all industries

    
</a></li>

                </ul>
              </div>
          </div>
         <div class="HeaderMenu-trailing-link rounded-bottom-2 flex-shrink-0 mt-lg-4 px-lg-4 py-4 py-lg-3 f5 text-semibold">
            <a href="https://github.com/solutions">
              View all solutions
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-chevron-right HeaderMenu-trailing-link-icon">
    <path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path>
</svg>
</a>         </div>
      </div>
</li>


                <li class="HeaderMenu-item position-relative flex-wrap flex-justify-between flex-items-center d-block d-lg-flex flex-lg-nowrap flex-lg-items-center js-details-container js-header-menu-item">
      <button type="button" class="HeaderMenu-link border-0 width-full width-lg-auto px-0 px-lg-2 py-lg-2 no-wrap d-flex flex-items-center flex-justify-between js-details-target" aria-expanded="false">
        Resources
        <svg opacity="0.5" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-chevron-down HeaderMenu-icon ml-1">
    <path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path>
</svg>
      </button>
      <div class="HeaderMenu-dropdown dropdown-menu rounded m-0 p-0 pt-2 pt-lg-4 position-relative position-lg-absolute left-0 left-lg-n3 pb-2 pb-lg-4 d-lg-flex flex-wrap dropdown-menu-wide">
          <div class="HeaderMenu-column px-lg-4 border-lg-right mb-4 mb-lg-0 pr-lg-7">
              <div class="border-bottom pb-3 pb-lg-0 border-lg-bottom-0">
                    <span class="d-block h4 color-fg-default my-1" id="resources-topics-heading">Topics</span>
                <ul class="list-style-none f5" aria-labelledby="resources-topics-heading">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;ai&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;ai_link_resources_navbar&quot;}" href="https://github.com/resources/articles/ai">
      AI

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;devops&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;devops_link_resources_navbar&quot;}" href="https://github.com/resources/articles/devops">
      DevOps

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;security&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;security_link_resources_navbar&quot;}" href="https://github.com/resources/articles/security">
      Security

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;software_development&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;software_development_link_resources_navbar&quot;}" href="https://github.com/resources/articles/software-development">
      Software Development

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;view_all&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;view_all_link_resources_navbar&quot;}" href="https://github.com/resources/articles">
      View all

    
</a></li>

                </ul>
              </div>
          </div>
          <div class="HeaderMenu-column px-lg-4">
              <div class="border-bottom pb-3 pb-lg-0 border-lg-bottom-0 border-bottom-0">
                    <span class="d-block h4 color-fg-default my-1" id="resources-explore-heading">Explore</span>
                <ul class="list-style-none f5" aria-labelledby="resources-explore-heading">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary Link--external" target="_blank" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;learning_pathways&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;learning_pathways_link_resources_navbar&quot;}" href="https://resources.github.com/learn/pathways">
      Learning Pathways

    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-link-external HeaderMenu-external-icon color-fg-subtle">
    <path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path>
</svg>
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary Link--external" target="_blank" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;events_amp_webinars&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;events_amp_webinars_link_resources_navbar&quot;}" href="https://resources.github.com/">
      Events &amp; Webinars

    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-link-external HeaderMenu-external-icon color-fg-subtle">
    <path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path>
</svg>
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;ebooks_amp_whitepapers&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;ebooks_amp_whitepapers_link_resources_navbar&quot;}" href="https://github.com/resources/whitepapers">
      Ebooks &amp; Whitepapers

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;customer_stories&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;customer_stories_link_resources_navbar&quot;}" href="https://github.com/customer-stories">
      Customer Stories

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary Link--external" target="_blank" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;partners&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;partners_link_resources_navbar&quot;}" href="https://partner.github.com/">
      Partners

    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-link-external HeaderMenu-external-icon color-fg-subtle">
    <path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path>
</svg>
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;executive_insights&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;executive_insights_link_resources_navbar&quot;}" href="https://github.com/solutions/executive-insights">
      Executive Insights

    
</a></li>

                </ul>
              </div>
          </div>
      </div>
</li>


                <li class="HeaderMenu-item position-relative flex-wrap flex-justify-between flex-items-center d-block d-lg-flex flex-lg-nowrap flex-lg-items-center js-details-container js-header-menu-item">
      <button type="button" class="HeaderMenu-link border-0 width-full width-lg-auto px-0 px-lg-2 py-lg-2 no-wrap d-flex flex-items-center flex-justify-between js-details-target" aria-expanded="false">
        Open Source
        <svg opacity="0.5" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-chevron-down HeaderMenu-icon ml-1">
    <path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path>
</svg>
      </button>
      <div class="HeaderMenu-dropdown dropdown-menu rounded m-0 p-0 pt-2 pt-lg-4 position-relative position-lg-absolute left-0 left-lg-n3 pb-2 pb-lg-4 px-lg-4">
          <div class="HeaderMenu-column">
              <div class="border-bottom pb-3 pb-lg-0 pb-lg-3 mb-3 mb-lg-0 mb-lg-3">
                <ul class="list-style-none f5">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;github_sponsors&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;github_sponsors_link_open_source_navbar&quot;}" href="https://github.com/sponsors">
      
      <div>
        <div class="color-fg-default h4">GitHub Sponsors</div>
        Fund open source developers
      </div>

    
</a></li>

                </ul>
              </div>
              <div class="border-bottom pb-3 pb-lg-0 pb-lg-3 mb-3 mb-lg-0 mb-lg-3">
                <ul class="list-style-none f5">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;the_readme_project&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;the_readme_project_link_open_source_navbar&quot;}" href="https://github.com/readme">
      
      <div>
        <div class="color-fg-default h4">The ReadME Project</div>
        GitHub community articles
      </div>

    
</a></li>

                </ul>
              </div>
              <div class="border-bottom pb-3 pb-lg-0 border-bottom-0">
                    <span class="d-block h4 color-fg-default my-1" id="open-source-repositories-heading">Repositories</span>
                <ul class="list-style-none f5" aria-labelledby="open-source-repositories-heading">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;topics&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;topics_link_open_source_navbar&quot;}" href="https://github.com/topics">
      Topics

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;trending&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;trending_link_open_source_navbar&quot;}" href="https://github.com/trending">
      Trending

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;collections&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;collections_link_open_source_navbar&quot;}" href="https://github.com/collections">
      Collections

    
</a></li>

                </ul>
              </div>
          </div>
      </div>
</li>


                <li class="HeaderMenu-item position-relative flex-wrap flex-justify-between flex-items-center d-block d-lg-flex flex-lg-nowrap flex-lg-items-center js-details-container js-header-menu-item">
      <button type="button" class="HeaderMenu-link border-0 width-full width-lg-auto px-0 px-lg-2 py-lg-2 no-wrap d-flex flex-items-center flex-justify-between js-details-target" aria-expanded="false">
        Enterprise
        <svg opacity="0.5" aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-chevron-down HeaderMenu-icon ml-1">
    <path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path>
</svg>
      </button>
      <div class="HeaderMenu-dropdown dropdown-menu rounded m-0 p-0 pt-2 pt-lg-4 position-relative position-lg-absolute left-0 left-lg-n3 pb-2 pb-lg-4 px-lg-4">
          <div class="HeaderMenu-column">
              <div class="border-bottom pb-3 pb-lg-0 pb-lg-3 mb-3 mb-lg-0 mb-lg-3">
                <ul class="list-style-none f5">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;enterprise_platform&quot;,&quot;context&quot;:&quot;enterprise&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;enterprise_platform_link_enterprise_navbar&quot;}" href="https://github.com/enterprise">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-stack color-fg-subtle mr-3">
    <path d="M11.063 1.456a1.749 1.749 0 0 1 1.874 0l8.383 5.316a1.751 1.751 0 0 1 0 2.956l-8.383 5.316a1.749 1.749 0 0 1-1.874 0L2.68 9.728a1.751 1.751 0 0 1 0-2.956Zm1.071 1.267a.25.25 0 0 0-.268 0L3.483 8.039a.25.25 0 0 0 0 .422l8.383 5.316a.25.25 0 0 0 .268 0l8.383-5.316a.25.25 0 0 0 0-.422Z"></path><path d="M1.867 12.324a.75.75 0 0 1 1.035-.232l8.964 5.685a.25.25 0 0 0 .268 0l8.964-5.685a.75.75 0 0 1 .804 1.267l-8.965 5.685a1.749 1.749 0 0 1-1.874 0l-8.965-5.685a.75.75 0 0 1-.231-1.035Z"></path><path d="M1.867 16.324a.75.75 0 0 1 1.035-.232l8.964 5.685a.25.25 0 0 0 .268 0l8.964-5.685a.75.75 0 0 1 .804 1.267l-8.965 5.685a1.749 1.749 0 0 1-1.874 0l-8.965-5.685a.75.75 0 0 1-.231-1.035Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">Enterprise platform</div>
        AI-powered developer platform
      </div>

    
</a></li>

                </ul>
              </div>
              <div class="border-bottom pb-3 pb-lg-0 border-bottom-0">
                    <span class="d-block h4 color-fg-default my-1" id="enterprise-available-add-ons-heading">Available add-ons</span>
                <ul class="list-style-none f5" aria-labelledby="enterprise-available-add-ons-heading">
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description pb-lg-3" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;github_advanced_security&quot;,&quot;context&quot;:&quot;enterprise&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;github_advanced_security_link_enterprise_navbar&quot;}" href="https://github.com/security/advanced-security">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-shield-check color-fg-subtle mr-3">
    <path d="M16.53 9.78a.75.75 0 0 0-1.06-1.06L11 13.19l-1.97-1.97a.75.75 0 0 0-1.06 1.06l2.5 2.5a.75.75 0 0 0 1.06 0l5-5Z"></path><path d="m12.54.637 8.25 2.675A1.75 1.75 0 0 1 22 4.976V10c0 6.19-3.771 10.704-9.401 12.83a1.704 1.704 0 0 1-1.198 0C5.77 20.705 2 16.19 2 10V4.976c0-.758.489-1.43 1.21-1.664L11.46.637a1.748 1.748 0 0 1 1.08 0Zm-.617 1.426-8.25 2.676a.249.249 0 0 0-.173.237V10c0 5.46 3.28 9.483 8.43 11.426a.199.199 0 0 0 .14 0C17.22 19.483 20.5 15.461 20.5 10V4.976a.25.25 0 0 0-.173-.237l-8.25-2.676a.253.253 0 0 0-.154 0Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">GitHub Advanced Security</div>
        Enterprise-grade security features
      </div>

    
</a></li>

                    
                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description pb-lg-3" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;copilot_for_business&quot;,&quot;context&quot;:&quot;enterprise&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;copilot_for_business_link_enterprise_navbar&quot;}" href="https://github.com/features/copilot/copilot-business">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-copilot color-fg-subtle mr-3">
    <path d="M23.922 16.992c-.861 1.495-5.859 5.023-11.922 5.023-6.063 0-11.061-3.528-11.922-5.023A.641.641 0 0 1 0 16.736v-2.869a.841.841 0 0 1 .053-.22c.372-.935 1.347-2.292 2.605-2.656.167-.429.414-1.055.644-1.517a10.195 10.195 0 0 1-.052-1.086c0-1.331.282-2.499 1.132-3.368.397-.406.89-.717 1.474-.952 1.399-1.136 3.392-2.093 6.122-2.093 2.731 0 4.767.957 6.166 2.093.584.235 1.077.546 1.474.952.85.869 1.132 2.037 1.132 3.368 0 .368-.014.733-.052 1.086.23.462.477 1.088.644 1.517 1.258.364 2.233 1.721 2.605 2.656a.832.832 0 0 1 .053.22v2.869a.641.641 0 0 1-.078.256ZM12.172 11h-.344a4.323 4.323 0 0 1-.355.508C10.703 12.455 9.555 13 7.965 13c-1.725 0-2.989-.359-3.782-1.259a2.005 2.005 0 0 1-.085-.104L4 11.741v6.585c1.435.779 4.514 2.179 8 2.179 3.486 0 6.565-1.4 8-2.179v-6.585l-.098-.104s-.033.045-.085.104c-.793.9-2.057 1.259-3.782 1.259-1.59 0-2.738-.545-3.508-1.492a4.323 4.323 0 0 1-.355-.508h-.016.016Zm.641-2.935c.136 1.057.403 1.913.878 2.497.442.544 1.134.938 2.344.938 1.573 0 2.292-.337 2.657-.751.384-.435.558-1.15.558-2.361 0-1.14-.243-1.847-.705-2.319-.477-.488-1.319-.862-2.824-1.025-1.487-.161-2.192.138-2.533.529-.269.307-.437.808-.438 1.578v.021c0 .265.021.562.063.893Zm-1.626 0c.042-.331.063-.628.063-.894v-.02c-.001-.77-.169-1.271-.438-1.578-.341-.391-1.046-.69-2.533-.529-1.505.163-2.347.537-2.824 1.025-.462.472-.705 1.179-.705 2.319 0 1.211.175 1.926.558 2.361.365.414 1.084.751 2.657.751 1.21 0 1.902-.394 2.344-.938.475-.584.742-1.44.878-2.497Z"></path><path d="M14.5 14.25a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1Zm-5 0a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">Copilot for business</div>
        Enterprise-grade AI features
      </div>

    
</a></li>

                    <li>
  <a class="HeaderMenu-dropdown-link d-block no-underline position-relative py-2 Link--secondary d-flex flex-items-center Link--has-description" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;premium_support&quot;,&quot;context&quot;:&quot;enterprise&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;premium_support_link_enterprise_navbar&quot;}" href="https://github.com/premium-support">
      <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-comment-discussion color-fg-subtle mr-3">
    <path d="M1.75 1h12.5c.966 0 1.75.784 1.75 1.75v9.5A1.75 1.75 0 0 1 14.25 14H8.061l-2.574 2.573A1.458 1.458 0 0 1 3 15.543V14H1.75A1.75 1.75 0 0 1 0 12.25v-9.5C0 1.784.784 1 1.75 1ZM1.5 2.75v9.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h6.5a.25.25 0 0 0 .25-.25v-9.5a.25.25 0 0 0-.25-.25H1.75a.25.25 0 0 0-.25.25Z"></path><path d="M22.5 8.75a.25.25 0 0 0-.25-.25h-3.5a.75.75 0 0 1 0-1.5h3.5c.966 0 1.75.784 1.75 1.75v9.5A1.75 1.75 0 0 1 22.25 20H21v1.543a1.457 1.457 0 0 1-2.487 1.03L15.939 20H10.75A1.75 1.75 0 0 1 9 18.25v-1.465a.75.75 0 0 1 1.5 0v1.465c0 .138.112.25.25.25h5.5a.75.75 0 0 1 .53.22l2.72 2.72v-2.19a.75.75 0 0 1 .75-.75h2a.25.25 0 0 0 .25-.25v-9.5Z"></path>
</svg>
      <div>
        <div class="color-fg-default h4">Premium Support</div>
        Enterprise-grade 24/7 support
      </div>

    
</a></li>

                </ul>
              </div>
          </div>
      </div>
</li>


                <li class="HeaderMenu-item position-relative flex-wrap flex-justify-between flex-items-center d-block d-lg-flex flex-lg-nowrap flex-lg-items-center js-details-container js-header-menu-item">
    <a class="HeaderMenu-link no-underline px-0 px-lg-2 py-3 py-lg-2 d-block d-lg-inline-block" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;pricing&quot;,&quot;context&quot;:&quot;global&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;pricing_link_global_navbar&quot;}" href="https://github.com/pricing">Pricing</a>
</li>

            </ul>
          </nav>

        <div class="d-flex flex-column flex-lg-row width-full flex-justify-end flex-lg-items-center text-center mt-3 mt-lg-0 text-lg-left ml-lg-3">
                


<qbsearch-input class="search-input" data-scope="repo:Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection" data-custom-scopes-path="/search/custom_scopes" data-delete-custom-scopes-csrf="6kNdvvU9GijyGtVScDYiYVXdkKTKW19WjjX7nrtmVh1AmiDnHLO-7Ko_QV95zubdBThDiLw19UV10HWhUHoBYA" data-max-custom-scopes="10" data-header-redesign-enabled="false" data-initial-value="" data-blackbird-suggestions-path="/search/suggestions" data-jump-to-suggestions-path="/_graphql/GetSuggestedNavigationDestinations" data-current-repository="Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection" data-current-org="" data-current-owner="Rafael-ZP" data-logged-in="false" data-copilot-chat-enabled="false" data-nl-search-enabled="false" data-retain-scroll-position="true" data-catalyst="">
  <div class="search-input-container search-with-dialog position-relative d-flex flex-row flex-items-center mr-4 rounded" data-action="click:qbsearch-input#searchInputContainerClicked">
      <button type="button" class="header-search-button placeholder input-button form-control d-flex flex-1 flex-self-stretch flex-items-center no-wrap width-full py-0 pl-2 pr-0 text-left border-0 box-shadow-none" data-target="qbsearch-input.inputButton" aria-label="Search or jump to…" aria-haspopup="dialog" placeholder="Search or jump to..." data-hotkey="s,/" autocapitalize="off" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;searchbar&quot;,&quot;context&quot;:&quot;global&quot;,&quot;tag&quot;:&quot;input&quot;,&quot;label&quot;:&quot;searchbar_input_global_navbar&quot;}" data-action="click:qbsearch-input#handleExpand">
        <div class="mr-2 color-fg-muted">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-search">
    <path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"></path>
</svg>
        </div>
        <span class="flex-1" data-target="qbsearch-input.inputButtonText">Search or jump to...</span>
          <div class="d-flex" data-target="qbsearch-input.hotkeyIndicator">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="20" aria-hidden="true" class="mr-1"><path fill="none" stroke="#979A9C" opacity=".4" d="M3.5.5h12c1.7 0 3 1.3 3 3v13c0 1.7-1.3 3-3 3h-12c-1.7 0-3-1.3-3-3v-13c0-1.7 1.3-3 3-3z"></path><path fill="#979A9C" d="M11.8 6L8 15.1h-.9L10.8 6h1z"></path></svg>

          </div>
      </button>

    <input type="hidden" name="type" class="js-site-search-type-field">

    
<div class="Overlay--hidden " data-modal-dialog-overlay="">
  <modal-dialog data-action="close:qbsearch-input#handleClose cancel:qbsearch-input#handleClose" data-target="qbsearch-input.searchSuggestionsDialog" role="dialog" id="search-suggestions-dialog" aria-modal="true" aria-labelledby="search-suggestions-dialog-header" data-view-component="true" class="Overlay Overlay--width-large Overlay--height-auto">
      <h1 id="search-suggestions-dialog-header" class="sr-only">Search code, repositories, users, issues, pull requests...</h1>
    <div class="Overlay-body Overlay-body--paddingNone">
      
          <div data-view-component="true">        <div class="search-suggestions position-fixed width-full color-shadow-large border color-fg-default color-bg-default overflow-hidden d-flex flex-column query-builder-container" style="border-radius: 12px;" data-target="qbsearch-input.queryBuilderContainer" hidden="">
          <!-- '"` --><!-- </textarea></xmp> --><form id="query-builder-test-form" action="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py" accept-charset="UTF-8" method="get">
  <query-builder data-target="qbsearch-input.queryBuilder" id="query-builder-query-builder-test" data-filter-key=":" data-view-component="true" class="QueryBuilder search-query-builder" data-min-width="300" data-catalyst="">
    <div class="FormControl FormControl--fullWidth">
      <label id="query-builder-test-label" for="query-builder-test" class="FormControl-label sr-only">
        Search
      </label>
      <div class="QueryBuilder-StyledInput width-fit " data-target="query-builder.styledInput">
          <span id="query-builder-test-leadingvisual-wrap" class="FormControl-input-leadingVisualWrap QueryBuilder-leadingVisualWrap">
            <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-search FormControl-input-leadingVisual">
    <path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"></path>
</svg>
          </span>
        <div data-target="query-builder.styledInputContainer" class="QueryBuilder-StyledInputContainer">
          <div aria-hidden="true" class="QueryBuilder-StyledInputContent" data-target="query-builder.styledInputContent"></div>
          <div class="QueryBuilder-InputWrapper">
            <div aria-hidden="true" class="QueryBuilder-Sizer" data-target="query-builder.sizer"><span></span></div>
            <input id="query-builder-test" name="query-builder-test" value="" autocomplete="off" type="text" role="combobox" spellcheck="false" aria-expanded="false" aria-describedby="validation-df2d690a-575c-47bd-a2ac-703f195118b0" data-target="query-builder.input" data-action="
          input:query-builder#inputChange
          blur:query-builder#inputBlur
          keydown:query-builder#inputKeydown
          focus:query-builder#inputFocus
        " data-view-component="true" class="FormControl-input QueryBuilder-Input FormControl-medium" aria-controls="query-builder-test-results" aria-autocomplete="list" aria-haspopup="listbox" style="width: 300px;">
          </div>
        </div>
          <span class="sr-only" id="query-builder-test-clear">Clear</span>
          <button role="button" id="query-builder-test-clear-button" aria-labelledby="query-builder-test-clear query-builder-test-label" data-target="query-builder.clearButton" data-action="
                click:query-builder#clear
                focus:query-builder#clearButtonFocus
                blur:query-builder#clearButtonBlur
              " variant="small" hidden="" type="button" data-view-component="true" class="Button Button--iconOnly Button--invisible Button--medium mr-1 px-2 py-0 d-flex flex-items-center rounded-1 color-fg-muted">  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x-circle-fill Button-visual">
    <path d="M2.343 13.657A8 8 0 1 1 13.658 2.343 8 8 0 0 1 2.343 13.657ZM6.03 4.97a.751.751 0 0 0-1.042.018.751.751 0 0 0-.018 1.042L6.94 8 4.97 9.97a.749.749 0 0 0 .326 1.275.749.749 0 0 0 .734-.215L8 9.06l1.97 1.97a.749.749 0 0 0 1.275-.326.749.749 0 0 0-.215-.734L9.06 8l1.97-1.97a.749.749 0 0 0-.326-1.275.749.749 0 0 0-.734.215L8 6.94Z"></path>
</svg>
</button>

      </div>
      <template id="search-icon"></template>

<template id="code-icon"></template>

<template id="file-code-icon"></template>

<template id="history-icon"></template>

<template id="repo-icon"></template>

<template id="bookmark-icon"></template>

<template id="plus-circle-icon"></template>

<template id="circle-icon"></template>

<template id="trash-icon"></template>

<template id="team-icon"></template>

<template id="project-icon"></template>

<template id="pencil-icon"></template>

<template id="copilot-icon"></template>

<template id="copilot-error-icon"></template>

<template id="workflow-icon"></template>

<template id="book-icon"></template>

<template id="code-review-icon"></template>

<template id="codespaces-icon"></template>

<template id="comment-icon"></template>

<template id="comment-discussion-icon"></template>

<template id="organization-icon"></template>

<template id="rocket-icon"></template>

<template id="shield-check-icon"></template>

<template id="heart-icon"></template>

<template id="server-icon"></template>

<template id="globe-icon"></template>

<template id="issue-opened-icon"></template>

<template id="device-mobile-icon"></template>

<template id="package-icon"></template>

<template id="credit-card-icon"></template>

<template id="play-icon"></template>

<template id="gift-icon"></template>

<template id="code-square-icon"></template>

<template id="device-desktop-icon"></template>

        <div class="position-relative">
                <ul role="listbox" class="ActionListWrap QueryBuilder-ListWrap" aria-label="Suggestions" data-action="
                    combobox-commit:query-builder#comboboxCommit
                    mousedown:query-builder#resultsMousedown
                  " data-target="query-builder.resultsList" data-persist-list="false" id="query-builder-test-results"></ul>
        </div>
      <div class="FormControl-inlineValidation" id="validation-df2d690a-575c-47bd-a2ac-703f195118b0" hidden="hidden">
        <span class="FormControl-inlineValidation--visual">
          <svg aria-hidden="true" height="12" viewBox="0 0 12 12" version="1.1" width="12" data-view-component="true" class="octicon octicon-alert-fill">
    <path d="M4.855.708c.5-.896 1.79-.896 2.29 0l4.675 8.351a1.312 1.312 0 0 1-1.146 1.954H1.33A1.313 1.313 0 0 1 .183 9.058ZM7 7V3H5v4Zm-1 3a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"></path>
</svg>
        </span>
        <span></span>
</div>    </div>
    <div data-target="query-builder.screenReaderFeedback" aria-live="polite" aria-atomic="true" class="sr-only"></div>
</query-builder></form>
          <div class="d-flex flex-row color-fg-muted px-3 text-small color-bg-default search-feedback-prompt">
            <a target="_blank" href="https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax" data-view-component="true" class="Link color-fg-accent text-normal ml-2">Search syntax tips</a>            <div class="d-flex flex-1"></div>
          </div>
        </div>
</div>

    </div>
</modal-dialog></div>
  </div>
  <div data-action="click:qbsearch-input#retract" class="dark-backdrop position-fixed" hidden="" data-target="qbsearch-input.darkBackdrop"></div>
  <div class="color-fg-default">
    
<dialog-helper>
  <dialog data-target="qbsearch-input.feedbackDialog" data-action="close:qbsearch-input#handleDialogClose cancel:qbsearch-input#handleDialogClose" id="feedback-dialog" aria-modal="true" aria-labelledby="feedback-dialog-title" aria-describedby="feedback-dialog-description" data-view-component="true" class="Overlay Overlay-whenNarrow Overlay--size-medium Overlay--motion-scaleFade Overlay--disableScroll">
    <div data-view-component="true" class="Overlay-header">
  <div class="Overlay-headerContentWrap">
    <div class="Overlay-titleWrap">
      <h1 class="Overlay-title " id="feedback-dialog-title">
        Provide feedback
      </h1>
        
    </div>
    <div class="Overlay-actionWrap">
      <button data-close-dialog-id="feedback-dialog" aria-label="Close" type="button" data-view-component="true" class="close-button Overlay-closeButton"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"></path>
</svg></button>
    </div>
  </div>
  
</div>
      <scrollable-region data-labelled-by="feedback-dialog-title" data-catalyst="" style="overflow: auto;">
        <div data-view-component="true" class="Overlay-body">        <!-- '"` --><!-- </textarea></xmp> --><form id="code-search-feedback-form" data-turbo="false" action="https://github.com/search/feedback" accept-charset="UTF-8" method="post"><input type="hidden" data-csrf="true" name="authenticity_token" value="oZrI/uflkk89zYtXgPNXiiVnwxkZ86WnSOi7GklgBuCODGIdL2wU6e9k1hFwLDalI/PYu2a/zbP3THDUCFBzDA==">
          <p>We read every piece of feedback, and take your input very seriously.</p>
          <textarea name="feedback" class="form-control width-full mb-2" style="height: 120px" id="feedback"></textarea>
          <input name="include_email" id="include_email" aria-label="Include my email address so I can be contacted" class="form-control mr-2" type="checkbox">
          <label for="include_email" style="font-weight: normal">Include my email address so I can be contacted</label>
</form></div>
      </scrollable-region>
      <div data-view-component="true" class="Overlay-footer Overlay-footer--alignEnd">          <button data-close-dialog-id="feedback-dialog" type="button" data-view-component="true" class="btn">    Cancel
</button>
          <button form="code-search-feedback-form" data-action="click:qbsearch-input#submitFeedback" type="submit" data-view-component="true" class="btn-primary btn">    Submit feedback
</button>
</div>
</dialog></dialog-helper>

    <custom-scopes data-target="qbsearch-input.customScopesManager" data-catalyst="">
    
<dialog-helper>
  <dialog data-target="custom-scopes.customScopesModalDialog" data-action="close:qbsearch-input#handleDialogClose cancel:qbsearch-input#handleDialogClose" id="custom-scopes-dialog" aria-modal="true" aria-labelledby="custom-scopes-dialog-title" aria-describedby="custom-scopes-dialog-description" data-view-component="true" class="Overlay Overlay-whenNarrow Overlay--size-medium Overlay--motion-scaleFade Overlay--disableScroll">
    <div data-view-component="true" class="Overlay-header Overlay-header--divided">
  <div class="Overlay-headerContentWrap">
    <div class="Overlay-titleWrap">
      <h1 class="Overlay-title " id="custom-scopes-dialog-title">
        Saved searches
      </h1>
        <h2 id="custom-scopes-dialog-description" class="Overlay-description">Use saved searches to filter your results more quickly</h2>
    </div>
    <div class="Overlay-actionWrap">
      <button data-close-dialog-id="custom-scopes-dialog" aria-label="Close" type="button" data-view-component="true" class="close-button Overlay-closeButton"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"></path>
</svg></button>
    </div>
  </div>
  
</div>
      <scrollable-region data-labelled-by="custom-scopes-dialog-title" data-catalyst="" style="overflow: auto;">
        <div data-view-component="true" class="Overlay-body">        <div data-target="custom-scopes.customScopesModalDialogFlash"></div>

        <div hidden="" class="create-custom-scope-form" data-target="custom-scopes.createCustomScopeForm">
        <!-- '"` --><!-- </textarea></xmp> --><form id="custom-scopes-dialog-form" data-turbo="false" action="https://github.com/search/custom_scopes" accept-charset="UTF-8" method="post"><input type="hidden" data-csrf="true" name="authenticity_token" value="ZH5s7RFoLLQPSUgZyGjpCCGqsOb2D/UUj1OsV2ArX7joAlM+ZNM+WgG9aRpDnmnLWu8C54v1WsjhIHpoT2Efaw==">
          <div data-target="custom-scopes.customScopesModalDialogFlash"></div>

          <input type="hidden" id="custom_scope_id" name="custom_scope_id" data-target="custom-scopes.customScopesIdField">

          <div class="form-group">
            <label for="custom_scope_name">Name</label>
            <auto-check src="/search/custom_scopes/check_name" required="" only-validate-on-blur="false">
              <input type="text" name="custom_scope_name" id="custom_scope_name" data-target="custom-scopes.customScopesNameField" class="form-control" autocomplete="off" placeholder="github-ruby" required="" maxlength="50" spellcheck="false">
              <input type="hidden" data-csrf="true" value="xfyXQG3N6mBZ+xz3kDVzMuUH655/O/FqcI1Z5/IqoVTBxj8M8PA4ivbOjLCU3+aeCBtzpIb8g2Ay9yIfzpA5Yg==">
            </auto-check>
          </div>

          <div class="form-group">
            <label for="custom_scope_query">Query</label>
            <input type="text" name="custom_scope_query" id="custom_scope_query" data-target="custom-scopes.customScopesQueryField" class="form-control" autocomplete="off" placeholder="(repo:mona/a OR repo:mona/b) AND lang:python" required="" maxlength="500">
          </div>

          <p class="text-small color-fg-muted">
            To see all available qualifiers, see our <a class="Link--inTextBlock" href="https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax">documentation</a>.
          </p>
</form>        </div>

        <div data-target="custom-scopes.manageCustomScopesForm">
          <div data-target="custom-scopes.list"></div>
        </div>

</div>
      </scrollable-region>
      <div data-view-component="true" class="Overlay-footer Overlay-footer--alignEnd Overlay-footer--divided">          <button data-action="click:custom-scopes#customScopesCancel" type="button" data-view-component="true" class="btn">    Cancel
</button>
          <button form="custom-scopes-dialog-form" data-action="click:custom-scopes#customScopesSubmit" data-target="custom-scopes.customScopesSubmitButton" type="submit" data-view-component="true" class="btn-primary btn">    Create saved search
</button>
</div>
</dialog></dialog-helper>
    </custom-scopes>
  </div>
</qbsearch-input>


            <div class="position-relative HeaderMenu-link-wrap d-lg-inline-block">
              <a href="https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2FRafael-ZP%2FLock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection%2Fblob%2Fmain%2FApp.py" class="HeaderMenu-link HeaderMenu-link--sign-in HeaderMenu-button flex-shrink-0 no-underline d-none d-lg-inline-flex border border-lg-0 rounded rounded-lg-0 px-2 py-1" style="margin-left: 12px;" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="e4964736d58e49e309c7e45dff1f30c1bdb8f1d0516061fe681b79be81e56de3" data-analytics-event="{&quot;category&quot;:&quot;Marketing nav&quot;,&quot;action&quot;:&quot;click to go to homepage&quot;,&quot;label&quot;:&quot;ref_page:Marketing;ref_cta:Sign in;ref_loc:Header&quot;}">
                Sign in
              </a>
            </div>

              <a href="https://github.com/signup?ref_cta=Sign+up&amp;ref_loc=header+logged+out&amp;ref_page=%2F%3Cuser-name%3E%2F%3Crepo-name%3E%2Fblob%2Fshow&amp;source=header-repo&amp;source_repo=Rafael-ZP%2FLock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection" class="HeaderMenu-link HeaderMenu-link--sign-up HeaderMenu-button flex-shrink-0 d-flex d-lg-inline-flex no-underline border color-border-default rounded px-2 py-1" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="e4964736d58e49e309c7e45dff1f30c1bdb8f1d0516061fe681b79be81e56de3" data-analytics-event="{&quot;category&quot;:&quot;Sign up&quot;,&quot;action&quot;:&quot;click to sign up for account&quot;,&quot;label&quot;:&quot;ref_page:/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show;ref_cta:Sign up;ref_loc:header logged out&quot;}">
                Sign up
              </a>

              
          <button type="button" class="sr-only js-header-menu-focus-trap d-block d-lg-none">Reseting focus</button>
        </div>
      </div>
    </div>
  </div>
</header>

      <div hidden="hidden" data-view-component="true" class="js-stale-session-flash stale-session-flash flash flash-warn flash-full">
  
        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-alert">
    <path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path>
</svg>
        <span class="js-stale-session-flash-signed-in" hidden="">You signed in with another tab or window. <a class="Link--inTextBlock" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py">Reload</a> to refresh your session.</span>
        <span class="js-stale-session-flash-signed-out" hidden="">You signed out in another tab or window. <a class="Link--inTextBlock" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py">Reload</a> to refresh your session.</span>
        <span class="js-stale-session-flash-switched" hidden="">You switched accounts on another tab or window. <a class="Link--inTextBlock" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py">Reload</a> to refresh your session.</span>

    <button id="icon-button-55dccfe6-9cb4-41d2-9b1c-49eb1291e96e" aria-labelledby="tooltip-2af2bb04-62ff-490e-9e69-a39e3970f839" type="button" data-view-component="true" class="Button Button--iconOnly Button--invisible Button--medium flash-close js-flash-close">  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x Button-visual">
    <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"></path>
</svg>
</button><tool-tip id="tooltip-2af2bb04-62ff-490e-9e69-a39e3970f839" for="icon-button-55dccfe6-9cb4-41d2-9b1c-49eb1291e96e" popover="manual" data-direction="s" data-type="label" data-view-component="true" class="sr-only position-absolute" aria-hidden="true" role="tooltip"><template shadowrootmode="open"><style>
      :host {
        --tooltip-top: var(--tool-tip-position-top, 0);
        --tooltip-left: var(--tool-tip-position-left, 0);
        padding: var(--overlay-paddingBlock-condensed) var(--overlay-padding-condensed) !important;
        font: var(--text-body-shorthand-small);
        color: var(--tooltip-fgColor, var(--fgColor-onEmphasis)) !important;
        text-align: center;
        text-decoration: none;
        text-shadow: none;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: break-word;
        white-space: pre;
        background: var(--tooltip-bgColor, var(--bgColor-emphasis)) !important;
        border-radius: var(--borderRadius-medium);
        border: 0 !important;
        opacity: 0;
        max-width: var(--overlay-width-small);
        word-wrap: break-word;
        white-space: normal;
        width: max-content !important;
        inset: var(--tooltip-top) auto auto var(--tooltip-left) !important;
        overflow: visible !important;
        text-wrap: balance;
      }

      :host(:is(.tooltip-n, .tooltip-nw, .tooltip-ne)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) - var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(:is(.tooltip-s, .tooltip-sw, .tooltip-se)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) + var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(.tooltip-w) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) - var(--overlay-offset, 0.25rem));
      }

      :host(.tooltip-e) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) + var(--overlay-offset, 0.25rem));
      }

      :host:after{
        position: absolute;
        display: block;
        right: 0;
        left: 0;
        height: var(--overlay-offset, 0.25rem);
        content: "";
      }

      :host(.tooltip-s):after,
      :host(.tooltip-se):after,
      :host(.tooltip-sw):after {
        bottom: 100%
      }

      :host(.tooltip-n):after,
      :host(.tooltip-ne):after,
      :host(.tooltip-nw):after {
        top: 100%;
      }

      @keyframes tooltip-appear {
        from {
          opacity: 0;
        }
        to {
          opacity: 1;
        }
      }

      :host(:popover-open),
      :host(:popover-open):before {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      :host(.\:popover-open) {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      @media (forced-colors: active) {
        :host {
          outline: solid 1px transparent;
        }

        :host:before {
          display: none;
        }
      }
    </style><slot></slot></template>Dismiss alert</tool-tip>


  
</div>
    </div>

  <div id="start-of-content" class="show-on-focus"></div>








    <div id="js-flash-container" class="flash-container" data-turbo-replace="">




  <template class="js-flash-template"></template>
</div>


    






  <div class="application-main " data-commit-hovercards-enabled="" data-discussion-hovercards-enabled="" data-issue-and-pr-hovercards-enabled="" data-project-hovercards-enabled="">
        <div itemscope="" itemtype="http://schema.org/SoftwareSourceCode" class="">
    <main id="js-repo-pjax-container">
      
      






  
  <div id="repository-container-header" class="pt-3 hide-full-screen" style="background-color: var(--page-header-bgColor, var(--color-page-header-bg));" data-turbo-replace="">

      <div class="d-flex flex-nowrap flex-justify-end mb-3  px-3 px-lg-5" style="gap: 1rem;">

        <div class="flex-auto min-width-0 width-fit">
            
  <div class=" d-flex flex-wrap flex-items-center wb-break-word f3 text-normal">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo color-fg-muted mr-2">
    <path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1h-8a1 1 0 0 0-1 1v6.708A2.486 2.486 0 0 1 4.5 9h8ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.249.249 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"></path>
</svg>
    
    <span class="author flex-self-stretch" itemprop="author">
      <a class="url fn" rel="author" data-hovercard-type="user" data-hovercard-url="/users/Rafael-ZP/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/Rafael-ZP">
        Rafael-ZP
</a>    </span>
    <span class="mx-1 flex-self-stretch color-fg-muted">/</span>
    <strong itemprop="name" class="mr-2 flex-self-stretch">
      <a data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection">Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection</a>
    </strong>

    <span></span><span class="Label Label--secondary v-align-middle mr-1">Public</span>
  </div>


        </div>

        <div id="repository-details-container" class="flex-shrink-0" data-turbo-replace="" style="max-width: 70%;">
            <ul class="pagehead-actions flex-shrink-0 d-none d-md-inline" style="padding: 2px 0;">
    
      

  <li>
            <a href="https://github.com/login?return_to=%2FRafael-ZP%2FLock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection" rel="nofollow" id="repository-details-watch-button" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;notification subscription menu watch&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="7524c7b7056ee98228d069c9751e6669085990d08dfca8cb8ebfbf03ea2bdca2" aria-label="You must be signed in to change notification settings" data-view-component="true" class="btn-sm btn" aria-describedby="tooltip-d2db94e8-c4d7-41d4-a249-9177125a5ad0">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-bell mr-2">
    <path d="M8 16a2 2 0 0 0 1.985-1.75c.017-.137-.097-.25-.235-.25h-3.5c-.138 0-.252.113-.235.25A2 2 0 0 0 8 16ZM3 5a5 5 0 0 1 10 0v2.947c0 .05.015.098.042.139l1.703 2.555A1.519 1.519 0 0 1 13.482 13H2.518a1.516 1.516 0 0 1-1.263-2.36l1.703-2.554A.255.255 0 0 0 3 7.947Zm5-3.5A3.5 3.5 0 0 0 4.5 5v2.947c0 .346-.102.683-.294.97l-1.703 2.556a.017.017 0 0 0-.003.01l.001.006c0 .002.002.004.004.006l.006.004.007.001h10.964l.007-.001.006-.004.004-.006.001-.007a.017.017 0 0 0-.003-.01l-1.703-2.554a1.745 1.745 0 0 1-.294-.97V5A3.5 3.5 0 0 0 8 1.5Z"></path>
</svg>Notifications
</a>    <tool-tip id="tooltip-d2db94e8-c4d7-41d4-a249-9177125a5ad0" for="repository-details-watch-button" popover="manual" data-direction="s" data-type="description" data-view-component="true" class="sr-only position-absolute" role="tooltip"><template shadowrootmode="open"><style>
      :host {
        --tooltip-top: var(--tool-tip-position-top, 0);
        --tooltip-left: var(--tool-tip-position-left, 0);
        padding: var(--overlay-paddingBlock-condensed) var(--overlay-padding-condensed) !important;
        font: var(--text-body-shorthand-small);
        color: var(--tooltip-fgColor, var(--fgColor-onEmphasis)) !important;
        text-align: center;
        text-decoration: none;
        text-shadow: none;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: break-word;
        white-space: pre;
        background: var(--tooltip-bgColor, var(--bgColor-emphasis)) !important;
        border-radius: var(--borderRadius-medium);
        border: 0 !important;
        opacity: 0;
        max-width: var(--overlay-width-small);
        word-wrap: break-word;
        white-space: normal;
        width: max-content !important;
        inset: var(--tooltip-top) auto auto var(--tooltip-left) !important;
        overflow: visible !important;
        text-wrap: balance;
      }

      :host(:is(.tooltip-n, .tooltip-nw, .tooltip-ne)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) - var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(:is(.tooltip-s, .tooltip-sw, .tooltip-se)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) + var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(.tooltip-w) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) - var(--overlay-offset, 0.25rem));
      }

      :host(.tooltip-e) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) + var(--overlay-offset, 0.25rem));
      }

      :host:after{
        position: absolute;
        display: block;
        right: 0;
        left: 0;
        height: var(--overlay-offset, 0.25rem);
        content: "";
      }

      :host(.tooltip-s):after,
      :host(.tooltip-se):after,
      :host(.tooltip-sw):after {
        bottom: 100%
      }

      :host(.tooltip-n):after,
      :host(.tooltip-ne):after,
      :host(.tooltip-nw):after {
        top: 100%;
      }

      @keyframes tooltip-appear {
        from {
          opacity: 0;
        }
        to {
          opacity: 1;
        }
      }

      :host(:popover-open),
      :host(:popover-open):before {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      :host(.\:popover-open) {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      @media (forced-colors: active) {
        :host {
          outline: solid 1px transparent;
        }

        :host:before {
          display: none;
        }
      }
    </style><slot></slot></template>You must be signed in to change notification settings</tool-tip>

  </li>

  <li>
          <a icon="repo-forked" id="fork-button" href="https://github.com/login?return_to=%2FRafael-ZP%2FLock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;repo details fork button&quot;,&quot;repository_id&quot;:963822950,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="fcdf5e4c21dec9a822ed3888f73634fe80a5823ee928f609b890cb33ee8aac46" data-view-component="true" class="btn-sm btn">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo-forked mr-2">
    <path d="M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-.878a2.25 2.25 0 1 1 1.5 0v.878a2.25 2.25 0 0 1-2.25 2.25h-1.5v2.128a2.251 2.251 0 1 1-1.5 0V8.5h-1.5A2.25 2.25 0 0 1 3.5 6.25v-.878a2.25 2.25 0 1 1 1.5 0ZM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Zm6.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm-3 8.75a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z"></path>
</svg>Fork
    <span id="repo-network-counter" data-pjax-replace="true" data-turbo-replace="true" title="0" data-view-component="true" class="Counter">0</span>
</a>
  </li>

  <li>
        <div data-view-component="true" class="BtnGroup d-flex">
        <a href="https://github.com/login?return_to=%2FRafael-ZP%2FLock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;star button&quot;,&quot;repository_id&quot;:963822950,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="a29748b332c5aba14ca5ceedbf89b0a35f1a53646124832981061a8442bf43e9" aria-label="You must be signed in to star a repository" data-view-component="true" class="tooltipped tooltipped-sw btn-sm btn">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star v-align-text-bottom d-inline-block mr-2">
    <path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.751.751 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Zm0 2.445L6.615 5.5a.75.75 0 0 1-.564.41l-3.097.45 2.24 2.184a.75.75 0 0 1 .216.664l-.528 3.084 2.769-1.456a.75.75 0 0 1 .698 0l2.77 1.456-.53-3.084a.75.75 0 0 1 .216-.664l2.24-2.183-3.096-.45a.75.75 0 0 1-.564-.41L8 2.694Z"></path>
</svg><span data-view-component="true" class="d-inline">
          Star
</span>          <span id="repo-stars-counter-star" aria-label="0 users starred this repository" data-singular-suffix="user starred this repository" data-plural-suffix="users starred this repository" data-turbo-replace="true" title="0" data-view-component="true" class="Counter js-social-count">0</span>
</a></div>
  </li>

</ul>

        </div>
      </div>

        <div id="responsive-meta-container" data-turbo-replace="">
</div>


          <nav data-pjax="#js-repo-pjax-container" aria-label="Repository" data-view-component="true" class="js-repo-nav js-sidenav-container-pjax js-responsive-underlinenav overflow-hidden UnderlineNav px-3 px-md-4 px-lg-5">

  <ul data-view-component="true" class="UnderlineNav-body list-style-none">
      <li data-view-component="true" class="d-inline-flex">
  <a id="code-tab" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection" data-tab-item="i0code-tab" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages repo_deployments repo_attestations /Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g c" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Code&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" aria-current="page" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item selected">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-code UnderlineNav-octicon d-none d-sm-inline">
    <path d="m11.28 3.22 4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734L13.94 8l-3.72-3.72a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215Zm-6.56 0a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L2.06 8l3.72 3.72a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L.47 8.53a.75.75 0 0 1 0-1.06Z"></path>
</svg>
        <span data-content="Code">Code</span>
          <span id="code-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="issues-tab" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/issues" data-tab-item="i1issues-tab" data-selected-links="repo_issues repo_labels repo_milestones /Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/issues" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g i" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Issues&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-opened UnderlineNav-octicon d-none d-sm-inline">
    <path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"></path>
</svg>
        <span data-content="Issues">Issues</span>
          <span id="issues-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="0" hidden="hidden" data-view-component="true" class="Counter">0</span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="pull-requests-tab" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/pulls" data-tab-item="i2pull-requests-tab" data-selected-links="repo_pulls checks /Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/pulls" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g p" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Pull requests&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-pull-request UnderlineNav-octicon d-none d-sm-inline">
    <path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.177L9.573.677A.25.25 0 0 1 10 .854V2.5h1A2.5 2.5 0 0 1 13.5 5v5.628a2.251 2.251 0 1 1-1.5 0V5a1 1 0 0 0-1-1h-1v1.646a.25.25 0 0 1-.427.177L7.177 3.427a.25.25 0 0 1 0-.354ZM3.75 2.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm0 9.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm8.25.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Z"></path>
</svg>
        <span data-content="Pull requests">Pull requests</span>
          <span id="pull-requests-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="0" hidden="hidden" data-view-component="true" class="Counter">0</span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="actions-tab" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/actions" data-tab-item="i3actions-tab" data-selected-links="repo_actions /Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/actions" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g a" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Actions&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-play UnderlineNav-octicon d-none d-sm-inline">
    <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"></path>
</svg>
        <span data-content="Actions">Actions</span>
          <span id="actions-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="projects-tab" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/projects" data-tab-item="i4projects-tab" data-selected-links="repo_projects new_repo_project repo_project /Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/projects" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g b" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Projects&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-table UnderlineNav-octicon d-none d-sm-inline">
    <path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25ZM6.5 6.5v8h7.75a.25.25 0 0 0 .25-.25V6.5Zm8-1.5V1.75a.25.25 0 0 0-.25-.25H6.5V5Zm-13 1.5v7.75c0 .138.112.25.25.25H5v-8ZM5 5V1.5H1.75a.25.25 0 0 0-.25.25V5Z"></path>
</svg>
        <span data-content="Projects">Projects</span>
          <span id="projects-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="0" hidden="hidden" data-view-component="true" class="Counter">0</span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="security-tab" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/security" data-tab-item="i5security-tab" data-selected-links="security overview alerts policy token_scanning code_scanning /Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/security" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g s" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Security&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-shield UnderlineNav-octicon d-none d-sm-inline">
    <path d="M7.467.133a1.748 1.748 0 0 1 1.066 0l5.25 1.68A1.75 1.75 0 0 1 15 3.48V7c0 1.566-.32 3.182-1.303 4.682-.983 1.498-2.585 2.813-5.032 3.855a1.697 1.697 0 0 1-1.33 0c-2.447-1.042-4.049-2.357-5.032-3.855C1.32 10.182 1 8.566 1 7V3.48a1.75 1.75 0 0 1 1.217-1.667Zm.61 1.429a.25.25 0 0 0-.153 0l-5.25 1.68a.25.25 0 0 0-.174.238V7c0 1.358.275 2.666 1.057 3.86.784 1.194 2.121 2.34 4.366 3.297a.196.196 0 0 0 .154 0c2.245-.956 3.582-2.104 4.366-3.298C13.225 9.666 13.5 8.36 13.5 7V3.48a.251.251 0 0 0-.174-.237l-5.25-1.68ZM8.75 4.75v3a.75.75 0 0 1-1.5 0v-3a.75.75 0 0 1 1.5 0ZM9 10.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path>
</svg>
        <span data-content="Security">Security</span>
          

    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="insights-tab" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/pulse" data-tab-item="i6insights-tab" data-selected-links="repo_graphs repo_contributors dependency_graph dependabot_updates pulse people community /Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/pulse" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Insights&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-graph UnderlineNav-octicon d-none d-sm-inline">
    <path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Zm14.28 2.53-5.25 5.25a.75.75 0 0 1-1.06 0L7 7.06 4.28 9.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.25-3.25a.75.75 0 0 1 1.06 0L10 7.94l4.72-4.72a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"></path>
</svg>
        <span data-content="Insights">Insights</span>
          <span id="insights-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
</ul>
    <div style="visibility:hidden;" data-view-component="true" class="UnderlineNav-actions js-responsive-underlinenav-overflow position-absolute pr-3 pr-md-4 pr-lg-5 right-0">      <action-menu data-select-variant="none" data-view-component="true" data-catalyst="" data-ready="true">
  <focus-group direction="vertical" mnemonics="" retain="">
    <button id="action-menu-c020e2b0-2d8e-4995-89b7-05798f4f8a81-button" popovertarget="action-menu-c020e2b0-2d8e-4995-89b7-05798f4f8a81-overlay" aria-controls="action-menu-c020e2b0-2d8e-4995-89b7-05798f4f8a81-list" aria-haspopup="true" aria-labelledby="tooltip-6654d5d5-f649-46af-b597-d226c5a03537" type="button" data-view-component="true" class="Button Button--iconOnly Button--secondary Button--medium UnderlineNav-item">  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-kebab-horizontal Button-visual">
    <path d="M8 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Zm13 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path>
</svg>
</button><tool-tip id="tooltip-6654d5d5-f649-46af-b597-d226c5a03537" for="action-menu-c020e2b0-2d8e-4995-89b7-05798f4f8a81-button" popover="manual" data-direction="s" data-type="label" data-view-component="true" class="sr-only position-absolute" aria-hidden="true" role="tooltip"><template shadowrootmode="open"><style>
      :host {
        --tooltip-top: var(--tool-tip-position-top, 0);
        --tooltip-left: var(--tool-tip-position-left, 0);
        padding: var(--overlay-paddingBlock-condensed) var(--overlay-padding-condensed) !important;
        font: var(--text-body-shorthand-small);
        color: var(--tooltip-fgColor, var(--fgColor-onEmphasis)) !important;
        text-align: center;
        text-decoration: none;
        text-shadow: none;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: break-word;
        white-space: pre;
        background: var(--tooltip-bgColor, var(--bgColor-emphasis)) !important;
        border-radius: var(--borderRadius-medium);
        border: 0 !important;
        opacity: 0;
        max-width: var(--overlay-width-small);
        word-wrap: break-word;
        white-space: normal;
        width: max-content !important;
        inset: var(--tooltip-top) auto auto var(--tooltip-left) !important;
        overflow: visible !important;
        text-wrap: balance;
      }

      :host(:is(.tooltip-n, .tooltip-nw, .tooltip-ne)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) - var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(:is(.tooltip-s, .tooltip-sw, .tooltip-se)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) + var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(.tooltip-w) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) - var(--overlay-offset, 0.25rem));
      }

      :host(.tooltip-e) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) + var(--overlay-offset, 0.25rem));
      }

      :host:after{
        position: absolute;
        display: block;
        right: 0;
        left: 0;
        height: var(--overlay-offset, 0.25rem);
        content: "";
      }

      :host(.tooltip-s):after,
      :host(.tooltip-se):after,
      :host(.tooltip-sw):after {
        bottom: 100%
      }

      :host(.tooltip-n):after,
      :host(.tooltip-ne):after,
      :host(.tooltip-nw):after {
        top: 100%;
      }

      @keyframes tooltip-appear {
        from {
          opacity: 0;
        }
        to {
          opacity: 1;
        }
      }

      :host(:popover-open),
      :host(:popover-open):before {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      :host(.\:popover-open) {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      @media (forced-colors: active) {
        :host {
          outline: solid 1px transparent;
        }

        :host:before {
          display: none;
        }
      }
    </style><slot></slot></template>Additional navigation options</tool-tip>


<anchored-position data-target="action-menu.overlay" id="action-menu-c020e2b0-2d8e-4995-89b7-05798f4f8a81-overlay" anchor="action-menu-c020e2b0-2d8e-4995-89b7-05798f4f8a81-button" align="start" side="outside-bottom" anchor-offset="normal" popover="auto" data-view-component="true" style="inset: 36px auto auto 0px;">
  <div data-view-component="true" class="Overlay Overlay--size-auto">
    
      <div data-view-component="true" class="Overlay-body Overlay-body--paddingNone">          <action-list data-catalyst="">
  <div data-view-component="true">
    <ul aria-labelledby="action-menu-c020e2b0-2d8e-4995-89b7-05798f4f8a81-button" id="action-menu-c020e2b0-2d8e-4995-89b7-05798f4f8a81-list" role="menu" data-view-component="true" class="ActionListWrap--inset ActionListWrap">
        <li hidden="" data-menu-item="i0code-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-c92246e7-6a38-4cb5-8cea-2596487f258b" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-code">
    <path d="m11.28 3.22 4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734L13.94 8l-3.72-3.72a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215Zm-6.56 0a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L2.06 8l3.72 3.72a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L.47 8.53a.75.75 0 0 1 0-1.06Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Code
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i1issues-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-8cafd59d-ca69-4f6a-a1e2-bf03d1fa42fa" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/issues" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-opened">
    <path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Issues
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i2pull-requests-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-85f921a8-ccef-4926-bc76-f2da82980fb6" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/pulls" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-pull-request">
    <path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.177L9.573.677A.25.25 0 0 1 10 .854V2.5h1A2.5 2.5 0 0 1 13.5 5v5.628a2.251 2.251 0 1 1-1.5 0V5a1 1 0 0 0-1-1h-1v1.646a.25.25 0 0 1-.427.177L7.177 3.427a.25.25 0 0 1 0-.354ZM3.75 2.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm0 9.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm8.25.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Pull requests
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i3actions-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-374cdc59-eb43-46f2-b9ff-9078b91c1f3c" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/actions" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-play">
    <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Actions
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i4projects-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-d9bdc6b2-9ccd-4f78-9ab8-03629699c1d5" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/projects" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-table">
    <path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25ZM6.5 6.5v8h7.75a.25.25 0 0 0 .25-.25V6.5Zm8-1.5V1.75a.25.25 0 0 0-.25-.25H6.5V5Zm-13 1.5v7.75c0 .138.112.25.25.25H5v-8ZM5 5V1.5H1.75a.25.25 0 0 0-.25.25V5Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Projects
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i5security-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-28d11644-51d9-450d-afcc-1b8295c7527c" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/security" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-shield">
    <path d="M7.467.133a1.748 1.748 0 0 1 1.066 0l5.25 1.68A1.75 1.75 0 0 1 15 3.48V7c0 1.566-.32 3.182-1.303 4.682-.983 1.498-2.585 2.813-5.032 3.855a1.697 1.697 0 0 1-1.33 0c-2.447-1.042-4.049-2.357-5.032-3.855C1.32 10.182 1 8.566 1 7V3.48a1.75 1.75 0 0 1 1.217-1.667Zm.61 1.429a.25.25 0 0 0-.153 0l-5.25 1.68a.25.25 0 0 0-.174.238V7c0 1.358.275 2.666 1.057 3.86.784 1.194 2.121 2.34 4.366 3.297a.196.196 0 0 0 .154 0c2.245-.956 3.582-2.104 4.366-3.298C13.225 9.666 13.5 8.36 13.5 7V3.48a.251.251 0 0 0-.174-.237l-5.25-1.68ZM8.75 4.75v3a.75.75 0 0 1-1.5 0v-3a.75.75 0 0 1 1.5 0ZM9 10.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Security
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i6insights-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-1ff5a4a7-79aa-4e2d-b512-c3a22ebb1625" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/pulse" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-graph">
    <path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Zm14.28 2.53-5.25 5.25a.75.75 0 0 1-1.06 0L7 7.06 4.28 9.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.25-3.25a.75.75 0 0 1 1.06 0L10 7.94l4.72-4.72a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Insights
</span>      
</a>
  
</li>
</ul>    
</div></action-list>


</div>
      
</div></anchored-position>  </focus-group>
</action-menu></div>
</nav>

  </div>

  



<turbo-frame id="repo-content-turbo-frame" target="_top" data-turbo-action="advance" class="">
    <div id="repo-content-pjax-container" class="repository-content ">
    



    
      
    








<react-app app-name="react-code-view" initial-path="/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py" style="display: block; min-height: calc(100vh - 64px);" data-attempted-ssr="true" data-ssr="true" data-lazy="false" data-alternate="false" data-data-router-enabled="false" data-catalyst="" class="loaded">
  
  <script type="application/json" data-target="react-app.embeddedData">{"payload":{"allShortcutsEnabled":false,"fileTree":{"":{"items":[{"name":"metrics","path":"metrics","contentType":"directory"},{"name":"models","path":"models","contentType":"directory"},{"name":"static","path":"static","contentType":"directory"},{"name":"templates","path":"templates","contentType":"directory"},{"name":"App.py","path":"App.py","contentType":"file"},{"name":"LICENSE","path":"LICENSE","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"},{"name":"Sample_2.mp4","path":"Sample_2.mp4","contentType":"file"},{"name":"Train_Blink.ipynb","path":"Train_Blink.ipynb","contentType":"file"},{"name":"Train_Gaze_Updated.ipynb","path":"Train_Gaze_Updated.ipynb","contentType":"file"},{"name":"requirements.txt","path":"requirements.txt","contentType":"file"}],"totalCount":11}},"fileTreeProcessingTime":3.4905359999999996,"foldersToFetch":[],"repo":{"id":963822950,"defaultBranch":"main","name":"Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection","ownerLogin":"Rafael-ZP","currentUserCanPush":false,"isFork":false,"isEmpty":false,"createdAt":"2025-04-10T09:05:04.000Z","ownerAvatar":"https://avatars.githubusercontent.com/u/104310982?v=4","public":true,"private":false,"isOrgOwned":false},"codeLineWrapEnabled":false,"symbolsExpanded":false,"treeExpanded":true,"refInfo":{"name":"main","listCacheKey":"v0:1744275904.0","canEdit":false,"refType":"branch","currentOid":"87aaaae0a1a4f1129e0412f949378a8a021e448b"},"path":"App.py","currentUser":null,"blob":{"rawLines":["import cv2","import dlib","import numpy as np","import time","from tensorflow.keras.models import load_model","from joblib import load as joblib_load","","# -----------------------------","# Load Haar cascades and Dlib model","# -----------------------------","face_cascade = cv2.CascadeClassifier(\"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/haarcascade_frontalface_default.xml\")","eye_cascade = cv2.CascadeClassifier(\"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/haarcascade_eye.xml\")","predictor_path = \"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/shape_predictor_68_face_landmarks.dat\"","detector = dlib.get_frontal_face_detector()","predictor = dlib.shape_predictor(predictor_path)","","# -----------------------------","# Load your two trained ML models","# -----------------------------","# 1. Fine tuned MobileNetV2 for looking / not looking","#    (expects full face color image resized to 224x224)","looking_model_path = \"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/fine_tuned_mobilenetv2.h5\"","looking_model = load_model(looking_model_path)","","# 2. Blink SVM for open/closed eye classification","#    (expects a flattened, normalized grayscale image of size 64x64, i.e. 4096 features)","blink_model_path = \"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/blink_svm.pkl\"","blink_model = joblib_load(blink_model_path)","","# -----------------------------","# Global parameters for attendance logic and fonts","# -----------------------------","ATTENDANCE_DURATION = 10       # seconds required to mark attendance (must be \"good\" for 10 sec)","ATTENDANCE_MESSAGE_DURATION = 2  # seconds to display the attendance message","MAX_CLOSED_FRAMES = 10         # if eyes are closed for \u003e this many consecutive frames, reset the timer","EYE_INPUT_SIZE = (64, 64)      # expected input size for the blink SVM (64x64 = 4096 features)","","# Font scales for on-screen text","FONT_SCALE = 1.2","ATTENDANCE_FONT_SCALE = 2.0","FONT_THICKNESS = 2","","def detect_face_eyes(video_source=0):","    \"\"\"","    Detects face, eyes, and landmarks; then runs the looking model and blink model.","    - The entire face region (in color) is passed to the looking model.","    - The eye regions (cropped from the grayscale image, expanded upward to include eyebrows)","      are resized to 64x64 and passed to the blink SVM.","      ","    Prediction logic (inverted if necessary):","      * For the looking model, a probability \u003c= 0.5 indicates \"looking.\"","      * For the blink model, a prediction of 1 indicates \"closed\" and 0 indicates \"open.\"","      ","    If a face is detected, the system checks if the subject is looking. If yes,","    it then checks the eyes. When both eyes are predicted as open (blink SVM returns 0)","    for a continuous period (ATTENDANCE_DURATION seconds) with only brief closures allowed,","    attendance is marked.","    \"\"\"","    cap = cv2.VideoCapture(video_source)","    if not cap.isOpened():","        print(\"Error: Unable to open video source.\")","        return","","    # Variables for attendance logic","    good_start_time = None       # Time when \"good\" criteria first started","    attendance_marked = False    # Whether attendance has been marked already","    attendance_marked_time = None  # Time at which attendance was marked (for display)","    closed_eye_consecutive = 0   # Count of consecutive frames with eyes closed","","    while cap.isOpened():","        ret, frame = cap.read()","        if not ret:","            break","","        current_time = time.time()","        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)","","        # -----------------------------","        # 1. Face detection using Haar cascade","        # -----------------------------","        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)","        if len(faces) \u003e 0:","            # For simplicity, select the largest detected face.","            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])","            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), FONT_THICKNESS)","","            # -----------------------------","            # 2. Looking detection using the entire face crop","            # -----------------------------","            face_roi = frame[y:y + h, x:x + w]","            try:","                face_roi_resized = cv2.resize(face_roi, (224, 224))","            except Exception as e:","                face_roi_resized = face_roi","            face_roi_norm = face_roi_resized.astype(\"float32\") / 255.0","            face_input = np.expand_dims(face_roi_norm, axis=0)","            looking_pred = looking_model.predict(face_input)","            # Inverted logic: probability \u003c= 0.5 means \"looking\"","            if looking_pred[0][0] \u003c= 0.5:","                looking = True","                looking_text = \"Looking: Yes\"","            else:","                looking = False","                looking_text = \"Looking: No\"","            cv2.putText(frame, looking_text, (x, y - 10),","                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 0), FONT_THICKNESS)","","            # -----------------------------","            # 3. Eye detection using Haar cascade within the face ROI","            # -----------------------------","            roi_gray = gray[y:y + h, x:x + w]","            roi_color = frame[y:y + h, x:x + w]","            eyes = eye_cascade.detectMultiScale(roi_gray)","            for (ex, ey, ew, eh) in eyes:","                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), FONT_THICKNESS)","","            # -----------------------------","            # 4. Get accurate eye coordinates using dlib’s landmarks","            # -----------------------------","            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))","            shape = predictor(gray, dlib_rect)","","            # Extract landmarks for left (36-41) and right (42-47) eyes.","            left_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])","            right_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])","","            for (ex_pt, ey_pt) in left_eye_pts:","                cv2.circle(frame, (ex_pt, ey_pt), 2, (0, 0, 255), -1)","            for (ex_pt, ey_pt) in right_eye_pts:","                cv2.circle(frame, (ex_pt, ey_pt), 2, (0, 0, 255), -1)","","            # Compute bounding boxes for each eye.","            lx, ly, lw, lh = cv2.boundingRect(left_eye_pts)","            rx, ry, rw, rh = cv2.boundingRect(right_eye_pts)","","            # Expand upward to include eyebrows (approx. 30% of the eye height)","            eyebrow_offset_left = int(0.3 * lh)","            eyebrow_offset_right = int(0.3 * rh)","            lx_new, ly_new = lx, max(ly - eyebrow_offset_left, 0)","            lw_new, lh_new = lw, lh + eyebrow_offset_left","            rx_new, ry_new = rx, max(ry - eyebrow_offset_right, 0)","            rw_new, rh_new = rw, rh + eyebrow_offset_right","","            cv2.rectangle(frame, (lx_new, ly_new), (lx_new + lw_new, ly_new + lh_new), (0, 255, 255), FONT_THICKNESS)","            cv2.rectangle(frame, (rx_new, ry_new), (rx_new + rw_new, ry_new + rh_new), (0, 255, 255), FONT_THICKNESS)","","            # -----------------------------","            # 5. Blink detection using the cropped eye images (grayscale with eyebrows)","            # -----------------------------","            left_eye_roi = gray[ly_new:ly_new + lh_new, lx_new:lx_new + lw_new]","            right_eye_roi = gray[ry_new:ry_new + rh_new, rx_new:rx_new + rw_new]","","            eyes_open = False","            eye_state = \"Unknown\"","            if left_eye_roi.size != 0 and right_eye_roi.size != 0:","                try:","                    left_eye_resized = cv2.resize(left_eye_roi, EYE_INPUT_SIZE)","                    right_eye_resized = cv2.resize(right_eye_roi, EYE_INPUT_SIZE)","                except Exception as e:","                    left_eye_resized = None","                    right_eye_resized = None","","                if left_eye_resized is not None and right_eye_resized is not None:","                    left_eye_input = left_eye_resized.flatten().astype(\"float32\") / 255.0","                    right_eye_input = right_eye_resized.flatten().astype(\"float32\") / 255.0","","                    left_pred = blink_model.predict([left_eye_input])[0]","                    right_pred = blink_model.predict([right_eye_input])[0]","","                    # Inverted logic: prediction of 1 indicates closed; 0 indicates open.","                    if left_pred == 1 and right_pred == 1:","                        eye_state = \"Closed\"","                        eyes_open = False","                    else:","                        eye_state = \"Open\"","                        eyes_open = True","","                    cv2.putText(frame, f\"Eye: {eye_state}\", (x, y + h + 20),","                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)","            else:","                eyes_open = False","","         ","            if looking:","                if eyes_open:","                    closed_eye_consecutive = 0","                    if good_start_time is None:","                        good_start_time = current_time","                else:","                    closed_eye_consecutive += 1","                    if closed_eye_consecutive \u003e MAX_CLOSED_FRAMES:","                        good_start_time = None","            else:","                good_start_time = None","                closed_eye_consecutive = 0","","            # If criteria are met for 10 seconds and attendance not yet marked.","            if good_start_time is not None:","                elapsed = current_time - good_start_time","                cv2.putText(frame, f\"Good for {elapsed:.1f}s\", (x, y - 40),","                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)","                if elapsed \u003e= ATTENDANCE_DURATION and not attendance_marked:","                    attendance_marked = True","                    attendance_marked_time = current_time","            else:","                cv2.putText(frame, \"Reset Timer\", (x, y - 40),","                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)","        else:","            good_start_time = None","            closed_eye_consecutive = 0","","        # -----------------------------","        # 7. Display the Attendance Marked message at the top-right (if within display duration)","        # -----------------------------","        if attendance_marked_time is not None:","            if current_time - attendance_marked_time \u003c= ATTENDANCE_MESSAGE_DURATION:","                # Calculate position at top right","                text = \"Attendance Marked\"","                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ATTENDANCE_FONT_SCALE, FONT_THICKNESS)","                # Place text with some margin from top-right corner","                pos = (frame.shape[1] - text_width - 20, text_height + 20)","                cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, ATTENDANCE_FONT_SCALE, (0, 255, 0), FONT_THICKNESS)","            else:","                # After 2 seconds, clear the attendance display time.","                attendance_marked_time = None","","        cv2.imshow(\"Face \u0026 Eye Detection with Attendance\", frame)","        if cv2.waitKey(1) \u0026 0xFF == ord('q'):","            break","","    cap.release()","    cv2.destroyAllWindows()","","# -----------------------------","# Main execution","# -----------------------------","if __name__ == \"__main__\":","    # Pass a video file path or 0 for webcam.","    detect_face_eyes(0)"],"stylingDirectives":[[[0,6,"pl-k"],[7,10,"pl-s1"]],[[0,6,"pl-k"],[7,11,"pl-s1"]],[[0,6,"pl-k"],[7,12,"pl-s1"],[13,15,"pl-k"],[16,18,"pl-s1"]],[[0,6,"pl-k"],[7,11,"pl-s1"]],[[0,4,"pl-k"],[5,15,"pl-s1"],[16,21,"pl-s1"],[22,28,"pl-s1"],[29,35,"pl-k"],[36,46,"pl-s1"]],[[0,4,"pl-k"],[5,11,"pl-s1"],[12,18,"pl-k"],[19,23,"pl-s1"],[24,26,"pl-k"],[27,38,"pl-s1"]],[],[[0,31,"pl-c"]],[[0,35,"pl-c"]],[[0,31,"pl-c"]],[[0,12,"pl-s1"],[13,14,"pl-c1"],[15,18,"pl-s1"],[19,36,"pl-c1"],[37,134,"pl-s"]],[[0,11,"pl-s1"],[12,13,"pl-c1"],[14,17,"pl-s1"],[18,35,"pl-c1"],[36,117,"pl-s"]],[[0,14,"pl-s1"],[15,16,"pl-c1"],[17,116,"pl-s"]],[[0,8,"pl-s1"],[9,10,"pl-c1"],[11,15,"pl-s1"],[16,41,"pl-c1"]],[[0,9,"pl-s1"],[10,11,"pl-c1"],[12,16,"pl-s1"],[17,32,"pl-c1"],[33,47,"pl-s1"]],[],[[0,31,"pl-c"]],[[0,33,"pl-c"]],[[0,31,"pl-c"]],[[0,53,"pl-c"]],[[0,55,"pl-c"]],[[0,18,"pl-s1"],[19,20,"pl-c1"],[21,108,"pl-s"]],[[0,13,"pl-s1"],[14,15,"pl-c1"],[16,26,"pl-en"],[27,45,"pl-s1"]],[],[[0,49,"pl-c"]],[[0,88,"pl-c"]],[[0,16,"pl-s1"],[17,18,"pl-c1"],[19,94,"pl-s"]],[[0,11,"pl-s1"],[12,13,"pl-c1"],[14,25,"pl-en"],[26,42,"pl-s1"]],[],[[0,31,"pl-c"]],[[0,50,"pl-c"]],[[0,31,"pl-c"]],[[0,19,"pl-c1"],[20,21,"pl-c1"],[22,24,"pl-c1"],[31,96,"pl-c"]],[[0,27,"pl-c1"],[28,29,"pl-c1"],[30,31,"pl-c1"],[33,76,"pl-c"]],[[0,17,"pl-c1"],[18,19,"pl-c1"],[20,22,"pl-c1"],[31,103,"pl-c"]],[[0,14,"pl-c1"],[15,16,"pl-c1"],[18,20,"pl-c1"],[22,24,"pl-c1"],[31,94,"pl-c"]],[],[[0,32,"pl-c"]],[[0,10,"pl-c1"],[11,12,"pl-c1"],[13,16,"pl-c1"]],[[0,21,"pl-c1"],[22,23,"pl-c1"],[24,27,"pl-c1"]],[[0,14,"pl-c1"],[15,16,"pl-c1"],[17,18,"pl-c1"]],[],[[0,3,"pl-k"],[4,20,"pl-en"],[21,33,"pl-s1"],[33,34,"pl-c1"],[34,35,"pl-c1"]],[[4,7,"pl-s"]],[[0,83,"pl-s"]],[[0,71,"pl-s"]],[[0,93,"pl-s"]],[[0,55,"pl-s"]],[[0,6,"pl-s"]],[[0,45,"pl-s"]],[[0,72,"pl-s"]],[[0,89,"pl-s"]],[[0,6,"pl-s"]],[[0,79,"pl-s"]],[[0,87,"pl-s"]],[[0,91,"pl-s"]],[[0,25,"pl-s"]],[[0,7,"pl-s"]],[[4,7,"pl-s1"],[8,9,"pl-c1"],[10,13,"pl-s1"],[14,26,"pl-c1"],[27,39,"pl-s1"]],[[4,6,"pl-k"],[7,10,"pl-c1"],[11,14,"pl-s1"],[15,23,"pl-c1"]],[[8,13,"pl-en"],[14,51,"pl-s"]],[[8,14,"pl-k"]],[],[[4,36,"pl-c"]],[[4,19,"pl-s1"],[20,21,"pl-c1"],[22,26,"pl-c1"],[33,74,"pl-c"]],[[4,21,"pl-s1"],[22,23,"pl-c1"],[24,29,"pl-c1"],[33,77,"pl-c"]],[[4,26,"pl-s1"],[27,28,"pl-c1"],[29,33,"pl-c1"],[35,86,"pl-c"]],[[4,26,"pl-s1"],[27,28,"pl-c1"],[29,30,"pl-c1"],[33,79,"pl-c"]],[],[[4,9,"pl-k"],[10,13,"pl-s1"],[14,22,"pl-c1"]],[[8,11,"pl-s1"],[13,18,"pl-s1"],[19,20,"pl-c1"],[21,24,"pl-s1"],[25,29,"pl-c1"]],[[8,10,"pl-k"],[11,14,"pl-c1"],[15,18,"pl-s1"]],[[12,17,"pl-k"]],[],[[8,20,"pl-s1"],[21,22,"pl-c1"],[23,27,"pl-s1"],[28,32,"pl-c1"]],[[8,12,"pl-s1"],[13,14,"pl-c1"],[15,18,"pl-s1"],[19,27,"pl-c1"],[28,33,"pl-s1"],[35,38,"pl-s1"],[39,53,"pl-c1"]],[],[[8,39,"pl-c"]],[[8,46,"pl-c"]],[[8,39,"pl-c"]],[[8,13,"pl-s1"],[14,15,"pl-c1"],[16,28,"pl-s1"],[29,45,"pl-c1"],[46,50,"pl-s1"],[52,63,"pl-s1"],[63,64,"pl-c1"],[64,67,"pl-c1"],[69,81,"pl-s1"],[81,82,"pl-c1"],[82,83,"pl-c1"]],[[8,10,"pl-k"],[11,14,"pl-en"],[15,20,"pl-s1"],[22,23,"pl-c1"],[24,25,"pl-c1"]],[[12,63,"pl-c"]],[[13,14,"pl-s1"],[16,17,"pl-s1"],[19,20,"pl-s1"],[22,23,"pl-s1"],[25,26,"pl-c1"],[27,30,"pl-en"],[31,36,"pl-s1"],[38,41,"pl-s1"],[41,42,"pl-c1"],[42,48,"pl-k"],[49,50,"pl-s1"],[52,53,"pl-s1"],[54,55,"pl-c1"],[57,58,"pl-c1"],[59,60,"pl-s1"],[61,62,"pl-c1"]],[[12,15,"pl-s1"],[16,25,"pl-c1"],[26,31,"pl-s1"],[34,35,"pl-s1"],[37,38,"pl-s1"],[42,43,"pl-s1"],[44,45,"pl-c1"],[46,47,"pl-s1"],[49,50,"pl-s1"],[51,52,"pl-c1"],[53,54,"pl-s1"],[58,61,"pl-c1"],[63,64,"pl-c1"],[66,67,"pl-c1"],[70,84,"pl-c1"]],[],[[12,43,"pl-c"]],[[12,61,"pl-c"]],[[12,43,"pl-c"]],[[12,20,"pl-s1"],[21,22,"pl-c1"],[23,28,"pl-s1"],[29,30,"pl-s1"],[31,32,"pl-s1"],[33,34,"pl-c1"],[35,36,"pl-s1"],[38,39,"pl-s1"],[40,41,"pl-s1"],[42,43,"pl-c1"],[44,45,"pl-s1"]],[[12,15,"pl-k"]],[[16,32,"pl-s1"],[33,34,"pl-c1"],[35,38,"pl-s1"],[39,45,"pl-c1"],[46,54,"pl-s1"],[57,60,"pl-c1"],[62,65,"pl-c1"]],[[12,18,"pl-k"],[19,28,"pl-v"],[29,31,"pl-k"],[32,33,"pl-s1"]],[[16,32,"pl-s1"],[33,34,"pl-c1"],[35,43,"pl-s1"]],[[12,25,"pl-s1"],[26,27,"pl-c1"],[28,44,"pl-s1"],[45,51,"pl-c1"],[52,61,"pl-s"],[63,64,"pl-c1"],[65,70,"pl-c1"]],[[12,22,"pl-s1"],[23,24,"pl-c1"],[25,27,"pl-s1"],[28,39,"pl-c1"],[40,53,"pl-s1"],[55,59,"pl-s1"],[59,60,"pl-c1"],[60,61,"pl-c1"]],[[12,24,"pl-s1"],[25,26,"pl-c1"],[27,40,"pl-s1"],[41,48,"pl-c1"],[49,59,"pl-s1"]],[[12,64,"pl-c"]],[[12,14,"pl-k"],[15,27,"pl-s1"],[28,29,"pl-c1"],[31,32,"pl-c1"],[34,36,"pl-c1"],[37,40,"pl-c1"]],[[16,23,"pl-s1"],[24,25,"pl-c1"],[26,30,"pl-c1"]],[[16,28,"pl-s1"],[29,30,"pl-c1"],[31,45,"pl-s"]],[[12,16,"pl-k"]],[[16,23,"pl-s1"],[24,25,"pl-c1"],[26,31,"pl-c1"]],[[16,28,"pl-s1"],[29,30,"pl-c1"],[31,44,"pl-s"]],[[12,15,"pl-s1"],[16,23,"pl-c1"],[24,29,"pl-s1"],[31,43,"pl-s1"],[46,47,"pl-s1"],[49,50,"pl-s1"],[51,52,"pl-c1"],[53,55,"pl-c1"]],[[24,27,"pl-s1"],[28,48,"pl-c1"],[50,60,"pl-c1"],[63,66,"pl-c1"],[68,69,"pl-c1"],[71,72,"pl-c1"],[75,89,"pl-c1"]],[],[[12,43,"pl-c"]],[[12,69,"pl-c"]],[[12,43,"pl-c"]],[[12,20,"pl-s1"],[21,22,"pl-c1"],[23,27,"pl-s1"],[28,29,"pl-s1"],[30,31,"pl-s1"],[32,33,"pl-c1"],[34,35,"pl-s1"],[37,38,"pl-s1"],[39,40,"pl-s1"],[41,42,"pl-c1"],[43,44,"pl-s1"]],[[12,21,"pl-s1"],[22,23,"pl-c1"],[24,29,"pl-s1"],[30,31,"pl-s1"],[32,33,"pl-s1"],[34,35,"pl-c1"],[36,37,"pl-s1"],[39,40,"pl-s1"],[41,42,"pl-s1"],[43,44,"pl-c1"],[45,46,"pl-s1"]],[[12,16,"pl-s1"],[17,18,"pl-c1"],[19,30,"pl-s1"],[31,47,"pl-c1"],[48,56,"pl-s1"]],[[12,15,"pl-k"],[17,19,"pl-s1"],[21,23,"pl-s1"],[25,27,"pl-s1"],[29,31,"pl-s1"],[33,35,"pl-c1"],[36,40,"pl-s1"]],[[16,19,"pl-s1"],[20,29,"pl-c1"],[30,39,"pl-s1"],[42,44,"pl-s1"],[46,48,"pl-s1"],[52,54,"pl-s1"],[55,56,"pl-c1"],[57,59,"pl-s1"],[61,63,"pl-s1"],[64,65,"pl-c1"],[66,68,"pl-s1"],[72,73,"pl-c1"],[75,78,"pl-c1"],[80,81,"pl-c1"],[84,98,"pl-c1"]],[],[[12,43,"pl-c"]],[[12,68,"pl-c"]],[[12,43,"pl-c"]],[[12,21,"pl-s1"],[22,23,"pl-c1"],[24,28,"pl-s1"],[29,38,"pl-c1"],[39,42,"pl-en"],[43,44,"pl-s1"],[47,50,"pl-en"],[51,52,"pl-s1"],[55,58,"pl-en"],[59,60,"pl-s1"],[61,62,"pl-c1"],[63,64,"pl-s1"],[67,70,"pl-en"],[71,72,"pl-s1"],[73,74,"pl-c1"],[75,76,"pl-s1"]],[[12,17,"pl-s1"],[18,19,"pl-c1"],[20,29,"pl-en"],[30,34,"pl-s1"],[36,45,"pl-s1"]],[],[[12,72,"pl-c"]],[[12,24,"pl-s1"],[25,26,"pl-c1"],[27,29,"pl-s1"],[30,35,"pl-c1"],[38,43,"pl-s1"],[44,48,"pl-c1"],[49,50,"pl-s1"],[52,53,"pl-c1"],[55,60,"pl-s1"],[61,65,"pl-c1"],[66,67,"pl-s1"],[69,70,"pl-c1"],[72,75,"pl-k"],[76,77,"pl-s1"],[78,80,"pl-c1"],[81,86,"pl-en"],[87,89,"pl-c1"],[91,93,"pl-c1"]],[[12,25,"pl-s1"],[26,27,"pl-c1"],[28,30,"pl-s1"],[31,36,"pl-c1"],[39,44,"pl-s1"],[45,49,"pl-c1"],[50,51,"pl-s1"],[53,54,"pl-c1"],[56,61,"pl-s1"],[62,66,"pl-c1"],[67,68,"pl-s1"],[70,71,"pl-c1"],[73,76,"pl-k"],[77,78,"pl-s1"],[79,81,"pl-c1"],[82,87,"pl-en"],[88,90,"pl-c1"],[92,94,"pl-c1"]],[],[[12,15,"pl-k"],[17,22,"pl-s1"],[24,29,"pl-s1"],[31,33,"pl-c1"],[34,46,"pl-s1"]],[[16,19,"pl-s1"],[20,26,"pl-c1"],[27,32,"pl-s1"],[35,40,"pl-s1"],[42,47,"pl-s1"],[50,51,"pl-c1"],[54,55,"pl-c1"],[57,58,"pl-c1"],[60,63,"pl-c1"],[66,67,"pl-c1"],[67,68,"pl-c1"]],[[12,15,"pl-k"],[17,22,"pl-s1"],[24,29,"pl-s1"],[31,33,"pl-c1"],[34,47,"pl-s1"]],[[16,19,"pl-s1"],[20,26,"pl-c1"],[27,32,"pl-s1"],[35,40,"pl-s1"],[42,47,"pl-s1"],[50,51,"pl-c1"],[54,55,"pl-c1"],[57,58,"pl-c1"],[60,63,"pl-c1"],[66,67,"pl-c1"],[67,68,"pl-c1"]],[],[[12,50,"pl-c"]],[[12,14,"pl-s1"],[16,18,"pl-s1"],[20,22,"pl-s1"],[24,26,"pl-s1"],[27,28,"pl-c1"],[29,32,"pl-s1"],[33,45,"pl-c1"],[46,58,"pl-s1"]],[[12,14,"pl-s1"],[16,18,"pl-s1"],[20,22,"pl-s1"],[24,26,"pl-s1"],[27,28,"pl-c1"],[29,32,"pl-s1"],[33,45,"pl-c1"],[46,59,"pl-s1"]],[],[[12,79,"pl-c"]],[[12,31,"pl-s1"],[32,33,"pl-c1"],[34,37,"pl-en"],[38,41,"pl-c1"],[42,43,"pl-c1"],[44,46,"pl-s1"]],[[12,32,"pl-s1"],[33,34,"pl-c1"],[35,38,"pl-en"],[39,42,"pl-c1"],[43,44,"pl-c1"],[45,47,"pl-s1"]],[[12,18,"pl-s1"],[20,26,"pl-s1"],[27,28,"pl-c1"],[29,31,"pl-s1"],[33,36,"pl-en"],[37,39,"pl-s1"],[40,41,"pl-c1"],[42,61,"pl-s1"],[63,64,"pl-c1"]],[[12,18,"pl-s1"],[20,26,"pl-s1"],[27,28,"pl-c1"],[29,31,"pl-s1"],[33,35,"pl-s1"],[36,37,"pl-c1"],[38,57,"pl-s1"]],[[12,18,"pl-s1"],[20,26,"pl-s1"],[27,28,"pl-c1"],[29,31,"pl-s1"],[33,36,"pl-en"],[37,39,"pl-s1"],[40,41,"pl-c1"],[42,62,"pl-s1"],[64,65,"pl-c1"]],[[12,18,"pl-s1"],[20,26,"pl-s1"],[27,28,"pl-c1"],[29,31,"pl-s1"],[33,35,"pl-s1"],[36,37,"pl-c1"],[38,58,"pl-s1"]],[],[[12,15,"pl-s1"],[16,25,"pl-c1"],[26,31,"pl-s1"],[34,40,"pl-s1"],[42,48,"pl-s1"],[52,58,"pl-s1"],[59,60,"pl-c1"],[61,67,"pl-s1"],[69,75,"pl-s1"],[76,77,"pl-c1"],[78,84,"pl-s1"],[88,89,"pl-c1"],[91,94,"pl-c1"],[96,99,"pl-c1"],[102,116,"pl-c1"]],[[12,15,"pl-s1"],[16,25,"pl-c1"],[26,31,"pl-s1"],[34,40,"pl-s1"],[42,48,"pl-s1"],[52,58,"pl-s1"],[59,60,"pl-c1"],[61,67,"pl-s1"],[69,75,"pl-s1"],[76,77,"pl-c1"],[78,84,"pl-s1"],[88,89,"pl-c1"],[91,94,"pl-c1"],[96,99,"pl-c1"],[102,116,"pl-c1"]],[],[[12,43,"pl-c"]],[[12,87,"pl-c"]],[[12,43,"pl-c"]],[[12,24,"pl-s1"],[25,26,"pl-c1"],[27,31,"pl-s1"],[32,38,"pl-s1"],[39,45,"pl-s1"],[46,47,"pl-c1"],[48,54,"pl-s1"],[56,62,"pl-s1"],[63,69,"pl-s1"],[70,71,"pl-c1"],[72,78,"pl-s1"]],[[12,25,"pl-s1"],[26,27,"pl-c1"],[28,32,"pl-s1"],[33,39,"pl-s1"],[40,46,"pl-s1"],[47,48,"pl-c1"],[49,55,"pl-s1"],[57,63,"pl-s1"],[64,70,"pl-s1"],[71,72,"pl-c1"],[73,79,"pl-s1"]],[],[[12,21,"pl-s1"],[22,23,"pl-c1"],[24,29,"pl-c1"]],[[12,21,"pl-s1"],[22,23,"pl-c1"],[24,33,"pl-s"]],[[12,14,"pl-k"],[15,27,"pl-s1"],[28,32,"pl-c1"],[33,35,"pl-c1"],[36,37,"pl-c1"],[38,41,"pl-c1"],[42,55,"pl-s1"],[56,60,"pl-c1"],[61,63,"pl-c1"],[64,65,"pl-c1"]],[[16,19,"pl-k"]],[[20,36,"pl-s1"],[37,38,"pl-c1"],[39,42,"pl-s1"],[43,49,"pl-c1"],[50,62,"pl-s1"],[64,78,"pl-c1"]],[[20,37,"pl-s1"],[38,39,"pl-c1"],[40,43,"pl-s1"],[44,50,"pl-c1"],[51,64,"pl-s1"],[66,80,"pl-c1"]],[[16,22,"pl-k"],[23,32,"pl-v"],[33,35,"pl-k"],[36,37,"pl-s1"]],[[20,36,"pl-s1"],[37,38,"pl-c1"],[39,43,"pl-c1"]],[[20,37,"pl-s1"],[38,39,"pl-c1"],[40,44,"pl-c1"]],[],[[16,18,"pl-k"],[19,35,"pl-s1"],[36,42,"pl-c1"],[36,38,"pl-c1"],[39,42,"pl-c1"],[43,47,"pl-c1"],[48,51,"pl-c1"],[52,69,"pl-s1"],[70,76,"pl-c1"],[70,72,"pl-c1"],[73,76,"pl-c1"],[77,81,"pl-c1"]],[[20,34,"pl-s1"],[35,36,"pl-c1"],[37,53,"pl-s1"],[54,61,"pl-c1"],[64,70,"pl-c1"],[71,80,"pl-s"],[82,83,"pl-c1"],[84,89,"pl-c1"]],[[20,35,"pl-s1"],[36,37,"pl-c1"],[38,55,"pl-s1"],[56,63,"pl-c1"],[66,72,"pl-c1"],[73,82,"pl-s"],[84,85,"pl-c1"],[86,91,"pl-c1"]],[],[[20,29,"pl-s1"],[30,31,"pl-c1"],[32,43,"pl-s1"],[44,51,"pl-c1"],[53,67,"pl-s1"],[70,71,"pl-c1"]],[[20,30,"pl-s1"],[31,32,"pl-c1"],[33,44,"pl-s1"],[45,52,"pl-c1"],[54,69,"pl-s1"],[72,73,"pl-c1"]],[],[[20,89,"pl-c"]],[[20,22,"pl-k"],[23,32,"pl-s1"],[33,35,"pl-c1"],[36,37,"pl-c1"],[38,41,"pl-c1"],[42,52,"pl-s1"],[53,55,"pl-c1"],[56,57,"pl-c1"]],[[24,33,"pl-s1"],[34,35,"pl-c1"],[36,44,"pl-s"]],[[24,33,"pl-s1"],[34,35,"pl-c1"],[36,41,"pl-c1"]],[[20,24,"pl-k"]],[[24,33,"pl-s1"],[34,35,"pl-c1"],[36,42,"pl-s"]],[[24,33,"pl-s1"],[34,35,"pl-c1"],[36,40,"pl-c1"]],[],[[20,23,"pl-s1"],[24,31,"pl-c1"],[32,37,"pl-s1"],[39,58,"pl-s"],[46,57,"pl-s1"],[46,47,"pl-kos"],[47,56,"pl-s1"],[56,57,"pl-kos"],[61,62,"pl-s1"],[64,65,"pl-s1"],[66,67,"pl-c1"],[68,69,"pl-s1"],[70,71,"pl-c1"],[72,74,"pl-c1"]],[[32,35,"pl-s1"],[36,56,"pl-c1"],[58,68,"pl-c1"],[71,72,"pl-c1"],[74,77,"pl-c1"],[79,82,"pl-c1"],[85,99,"pl-c1"]],[[12,16,"pl-k"]],[[16,25,"pl-s1"],[26,27,"pl-c1"],[28,33,"pl-c1"]],[],[],[[12,14,"pl-k"],[15,22,"pl-s1"]],[[16,18,"pl-k"],[19,28,"pl-s1"]],[[20,42,"pl-s1"],[43,44,"pl-c1"],[45,46,"pl-c1"]],[[20,22,"pl-k"],[23,38,"pl-s1"],[39,41,"pl-c1"],[42,46,"pl-c1"]],[[24,39,"pl-s1"],[40,41,"pl-c1"],[42,54,"pl-s1"]],[[16,20,"pl-k"]],[[20,42,"pl-s1"],[43,45,"pl-c1"],[46,47,"pl-c1"]],[[20,22,"pl-k"],[23,45,"pl-s1"],[46,47,"pl-c1"],[48,65,"pl-c1"]],[[24,39,"pl-s1"],[40,41,"pl-c1"],[42,46,"pl-c1"]],[[12,16,"pl-k"]],[[16,31,"pl-s1"],[32,33,"pl-c1"],[34,38,"pl-c1"]],[[16,38,"pl-s1"],[39,40,"pl-c1"],[41,42,"pl-c1"]],[],[[12,79,"pl-c"]],[[12,14,"pl-k"],[15,30,"pl-s1"],[31,37,"pl-c1"],[31,33,"pl-c1"],[34,37,"pl-c1"],[38,42,"pl-c1"]],[[16,23,"pl-s1"],[24,25,"pl-c1"],[26,38,"pl-s1"],[39,40,"pl-c1"],[41,56,"pl-s1"]],[[16,19,"pl-s1"],[20,27,"pl-c1"],[28,33,"pl-s1"],[35,61,"pl-s"],[46,59,"pl-s1"],[46,47,"pl-kos"],[47,54,"pl-s1"],[58,59,"pl-kos"],[64,65,"pl-s1"],[67,68,"pl-s1"],[69,70,"pl-c1"],[71,73,"pl-c1"]],[[28,31,"pl-s1"],[32,52,"pl-c1"],[54,64,"pl-c1"],[67,68,"pl-c1"],[70,73,"pl-c1"],[75,76,"pl-c1"],[79,93,"pl-c1"]],[[16,18,"pl-k"],[19,26,"pl-s1"],[27,29,"pl-c1"],[30,49,"pl-c1"],[50,53,"pl-c1"],[54,57,"pl-c1"],[58,75,"pl-s1"]],[[20,37,"pl-s1"],[38,39,"pl-c1"],[40,44,"pl-c1"]],[[20,42,"pl-s1"],[43,44,"pl-c1"],[45,57,"pl-s1"]],[[12,16,"pl-k"]],[[16,19,"pl-s1"],[20,27,"pl-c1"],[28,33,"pl-s1"],[35,48,"pl-s"],[51,52,"pl-s1"],[54,55,"pl-s1"],[56,57,"pl-c1"],[58,60,"pl-c1"]],[[28,31,"pl-s1"],[32,52,"pl-c1"],[54,64,"pl-c1"],[67,68,"pl-c1"],[70,71,"pl-c1"],[73,76,"pl-c1"],[79,93,"pl-c1"]],[[8,12,"pl-k"]],[[12,27,"pl-s1"],[28,29,"pl-c1"],[30,34,"pl-c1"]],[[12,34,"pl-s1"],[35,36,"pl-c1"],[37,38,"pl-c1"]],[],[[8,39,"pl-c"]],[[8,96,"pl-c"]],[[8,39,"pl-c"]],[[8,10,"pl-k"],[11,33,"pl-s1"],[34,40,"pl-c1"],[34,36,"pl-c1"],[37,40,"pl-c1"],[41,45,"pl-c1"]],[[12,14,"pl-k"],[15,27,"pl-s1"],[28,29,"pl-c1"],[30,52,"pl-s1"],[53,55,"pl-c1"],[56,83,"pl-c1"]],[[16,49,"pl-c"]],[[16,20,"pl-s1"],[21,22,"pl-c1"],[23,42,"pl-s"]],[[17,27,"pl-s1"],[29,40,"pl-s1"],[43,44,"pl-s1"],[45,46,"pl-c1"],[47,50,"pl-s1"],[51,62,"pl-c1"],[63,67,"pl-s1"],[69,72,"pl-s1"],[73,93,"pl-c1"],[95,116,"pl-c1"],[118,132,"pl-c1"]],[[16,67,"pl-c"]],[[16,19,"pl-s1"],[20,21,"pl-c1"],[23,28,"pl-s1"],[29,34,"pl-c1"],[35,36,"pl-c1"],[38,39,"pl-c1"],[40,50,"pl-s1"],[51,52,"pl-c1"],[53,55,"pl-c1"],[57,68,"pl-s1"],[69,70,"pl-c1"],[71,73,"pl-c1"]],[[16,19,"pl-s1"],[20,27,"pl-c1"],[28,33,"pl-s1"],[35,39,"pl-s1"],[41,44,"pl-s1"],[46,49,"pl-s1"],[50,70,"pl-c1"],[72,93,"pl-c1"],[96,97,"pl-c1"],[99,102,"pl-c1"],[104,105,"pl-c1"],[108,122,"pl-c1"]],[[12,16,"pl-k"]],[[16,69,"pl-c"]],[[16,38,"pl-s1"],[39,40,"pl-c1"],[41,45,"pl-c1"]],[],[[8,11,"pl-s1"],[12,18,"pl-c1"],[19,57,"pl-s"],[59,64,"pl-s1"]],[[8,10,"pl-k"],[11,14,"pl-s1"],[15,22,"pl-c1"],[23,24,"pl-c1"],[26,27,"pl-c1"],[28,32,"pl-c1"],[33,35,"pl-c1"],[36,39,"pl-en"],[40,43,"pl-s"]],[[12,17,"pl-k"]],[],[[4,7,"pl-s1"],[8,15,"pl-c1"]],[[4,7,"pl-s1"],[8,25,"pl-c1"]],[],[[0,31,"pl-c"]],[[0,16,"pl-c"]],[[0,31,"pl-c"]],[[0,2,"pl-k"],[3,11,"pl-s1"],[12,14,"pl-c1"],[15,25,"pl-s"]],[[4,45,"pl-c"]],[[4,20,"pl-en"],[21,22,"pl-c1"]]],"colorizedLines":null,"csv":null,"csvError":null,"dependabotInfo":{"showConfigurationBanner":false,"configFilePath":null,"networkDependabotPath":"/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/network/updates","dismissConfigurationNoticePath":"/settings/dismiss-notice/dependabot_configuration_notice","configurationNoticeDismissed":null},"displayName":"App.py","displayUrl":"https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py?raw=true","headerInfo":{"blobSize":"11.1 KB","deleteTooltip":"You must be signed in to make or propose changes","editTooltip":"You must be signed in to make or propose changes","ghDesktopPath":"https://desktop.github.com","isGitLfs":false,"onBranch":true,"shortPath":"870e4c8","siteNavLoginPath":"/login?return_to=https%3A%2F%2Fgithub.com%2FRafael-ZP%2FLock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection%2Fblob%2Fmain%2FApp.py","isCSV":false,"isRichtext":false,"toc":null,"lineInfo":{"truncatedLoc":"239","truncatedSloc":"206"},"mode":"file"},"image":false,"isCodeownersFile":null,"isPlain":false,"isValidLegacyIssueTemplate":false,"issueTemplate":null,"discussionTemplate":null,"language":"Python","languageID":303,"large":false,"planSupportInfo":{"repoIsFork":null,"repoOwnedByCurrentUser":null,"requestFullPath":"/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/blob/main/App.py","showFreeOrgGatedFeatureMessage":null,"showPlanSupportBanner":null,"upgradeDataAttributes":null,"upgradePath":null},"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_dockerfile","releasePath":"/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/releases/new?marketplace=true","showPublishActionBanner":false},"rawBlobUrl":"https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/raw/refs/heads/main/App.py","renderImageOrRaw":false,"richText":null,"renderedFileInfo":null,"shortPath":null,"symbolsEnabled":true,"tabSize":8,"topBannersInfo":{"overridingGlobalFundingFile":false,"globalPreferredFundingPath":null,"showInvalidCitationWarning":false,"citationHelpUrl":"https://docs.github.com/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files","actionsOnboardingTip":null},"truncated":false,"viewable":true,"workflowRedirectUrl":null,"symbols":{"timed_out":false,"not_analyzed":false,"symbols":[{"name":"face_cascade","kind":"constant","ident_start":241,"ident_end":253,"extent_start":241,"extent_end":376,"fully_qualified_name":"face_cascade","ident_utf16":{"start":{"line_number":10,"utf16_col":0},"end":{"line_number":10,"utf16_col":12}},"extent_utf16":{"start":{"line_number":10,"utf16_col":0},"end":{"line_number":10,"utf16_col":135}}},{"name":"eye_cascade","kind":"constant","ident_start":377,"ident_end":388,"extent_start":377,"extent_end":495,"fully_qualified_name":"eye_cascade","ident_utf16":{"start":{"line_number":11,"utf16_col":0},"end":{"line_number":11,"utf16_col":11}},"extent_utf16":{"start":{"line_number":11,"utf16_col":0},"end":{"line_number":11,"utf16_col":118}}},{"name":"predictor_path","kind":"constant","ident_start":496,"ident_end":510,"extent_start":496,"extent_end":612,"fully_qualified_name":"predictor_path","ident_utf16":{"start":{"line_number":12,"utf16_col":0},"end":{"line_number":12,"utf16_col":14}},"extent_utf16":{"start":{"line_number":12,"utf16_col":0},"end":{"line_number":12,"utf16_col":116}}},{"name":"detector","kind":"constant","ident_start":613,"ident_end":621,"extent_start":613,"extent_end":656,"fully_qualified_name":"detector","ident_utf16":{"start":{"line_number":13,"utf16_col":0},"end":{"line_number":13,"utf16_col":8}},"extent_utf16":{"start":{"line_number":13,"utf16_col":0},"end":{"line_number":13,"utf16_col":43}}},{"name":"predictor","kind":"constant","ident_start":657,"ident_end":666,"extent_start":657,"extent_end":705,"fully_qualified_name":"predictor","ident_utf16":{"start":{"line_number":14,"utf16_col":0},"end":{"line_number":14,"utf16_col":9}},"extent_utf16":{"start":{"line_number":14,"utf16_col":0},"end":{"line_number":14,"utf16_col":48}}},{"name":"looking_model_path","kind":"constant","ident_start":915,"ident_end":933,"extent_start":915,"extent_end":1023,"fully_qualified_name":"looking_model_path","ident_utf16":{"start":{"line_number":21,"utf16_col":0},"end":{"line_number":21,"utf16_col":18}},"extent_utf16":{"start":{"line_number":21,"utf16_col":0},"end":{"line_number":21,"utf16_col":108}}},{"name":"looking_model","kind":"constant","ident_start":1024,"ident_end":1037,"extent_start":1024,"extent_end":1070,"fully_qualified_name":"looking_model","ident_utf16":{"start":{"line_number":22,"utf16_col":0},"end":{"line_number":22,"utf16_col":13}},"extent_utf16":{"start":{"line_number":22,"utf16_col":0},"end":{"line_number":22,"utf16_col":46}}},{"name":"blink_model_path","kind":"constant","ident_start":1211,"ident_end":1227,"extent_start":1211,"extent_end":1305,"fully_qualified_name":"blink_model_path","ident_utf16":{"start":{"line_number":26,"utf16_col":0},"end":{"line_number":26,"utf16_col":16}},"extent_utf16":{"start":{"line_number":26,"utf16_col":0},"end":{"line_number":26,"utf16_col":94}}},{"name":"blink_model","kind":"constant","ident_start":1306,"ident_end":1317,"extent_start":1306,"extent_end":1349,"fully_qualified_name":"blink_model","ident_utf16":{"start":{"line_number":27,"utf16_col":0},"end":{"line_number":27,"utf16_col":11}},"extent_utf16":{"start":{"line_number":27,"utf16_col":0},"end":{"line_number":27,"utf16_col":43}}},{"name":"ATTENDANCE_DURATION","kind":"constant","ident_start":1466,"ident_end":1485,"extent_start":1466,"extent_end":1490,"fully_qualified_name":"ATTENDANCE_DURATION","ident_utf16":{"start":{"line_number":32,"utf16_col":0},"end":{"line_number":32,"utf16_col":19}},"extent_utf16":{"start":{"line_number":32,"utf16_col":0},"end":{"line_number":32,"utf16_col":24}}},{"name":"ATTENDANCE_MESSAGE_DURATION","kind":"constant","ident_start":1563,"ident_end":1590,"extent_start":1563,"extent_end":1594,"fully_qualified_name":"ATTENDANCE_MESSAGE_DURATION","ident_utf16":{"start":{"line_number":33,"utf16_col":0},"end":{"line_number":33,"utf16_col":27}},"extent_utf16":{"start":{"line_number":33,"utf16_col":0},"end":{"line_number":33,"utf16_col":31}}},{"name":"MAX_CLOSED_FRAMES","kind":"constant","ident_start":1640,"ident_end":1657,"extent_start":1640,"extent_end":1662,"fully_qualified_name":"MAX_CLOSED_FRAMES","ident_utf16":{"start":{"line_number":34,"utf16_col":0},"end":{"line_number":34,"utf16_col":17}},"extent_utf16":{"start":{"line_number":34,"utf16_col":0},"end":{"line_number":34,"utf16_col":22}}},{"name":"EYE_INPUT_SIZE","kind":"constant","ident_start":1744,"ident_end":1758,"extent_start":1744,"extent_end":1769,"fully_qualified_name":"EYE_INPUT_SIZE","ident_utf16":{"start":{"line_number":35,"utf16_col":0},"end":{"line_number":35,"utf16_col":14}},"extent_utf16":{"start":{"line_number":35,"utf16_col":0},"end":{"line_number":35,"utf16_col":25}}},{"name":"FONT_SCALE","kind":"constant","ident_start":1873,"ident_end":1883,"extent_start":1873,"extent_end":1889,"fully_qualified_name":"FONT_SCALE","ident_utf16":{"start":{"line_number":38,"utf16_col":0},"end":{"line_number":38,"utf16_col":10}},"extent_utf16":{"start":{"line_number":38,"utf16_col":0},"end":{"line_number":38,"utf16_col":16}}},{"name":"ATTENDANCE_FONT_SCALE","kind":"constant","ident_start":1890,"ident_end":1911,"extent_start":1890,"extent_end":1917,"fully_qualified_name":"ATTENDANCE_FONT_SCALE","ident_utf16":{"start":{"line_number":39,"utf16_col":0},"end":{"line_number":39,"utf16_col":21}},"extent_utf16":{"start":{"line_number":39,"utf16_col":0},"end":{"line_number":39,"utf16_col":27}}},{"name":"FONT_THICKNESS","kind":"constant","ident_start":1918,"ident_end":1932,"extent_start":1918,"extent_end":1936,"fully_qualified_name":"FONT_THICKNESS","ident_utf16":{"start":{"line_number":40,"utf16_col":0},"end":{"line_number":40,"utf16_col":14}},"extent_utf16":{"start":{"line_number":40,"utf16_col":0},"end":{"line_number":40,"utf16_col":18}}},{"name":"detect_face_eyes","kind":"function","ident_start":1942,"ident_end":1958,"extent_start":1938,"extent_end":11198,"fully_qualified_name":"detect_face_eyes","ident_utf16":{"start":{"line_number":42,"utf16_col":4},"end":{"line_number":42,"utf16_col":20}},"extent_utf16":{"start":{"line_number":42,"utf16_col":0},"end":{"line_number":231,"utf16_col":27}}}]}},"copilotInfo":null,"copilotAccessAllowed":false,"modelsAccessAllowed":false,"modelsRepoIntegrationEnabled":false,"csrf_tokens":{"/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/branches":{"post":"bjCpJjGpVikQRR5gD8kFM4g-PtWpTT7aKqVtIsHgXNckOtqvJDRQgqJSQkDWaKwSNLxbBS0wwLgJTxQ194GUdg"},"/repos/preferences":{"post":"5whxTyvm0O75hjJ2c3_oHI2HTikesmXDOqr9zUsBMrVFckiHJvKJDbioC38iLxztmqsKhrC7zssumT5uMNWBmw"}}},"title":"Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/App.py at main · Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection","appPayload":{"helpUrl":"https://docs.github.com","findFileWorkerPath":"/assets-cdn/worker/find-file-worker-7d7eb7c71814.js","findInFileWorkerPath":"/assets-cdn/worker/find-in-file-worker-708ec8ade250.js","githubDevUrl":null,"enabled_features":{"code_nav_ui_events":false,"overview_shared_code_dropdown_button":true,"react_blob_overlay":false,"accessible_code_button":true,"github_models_repo_integration":false}}}</script>
  <div data-target="react-app.reactRoot"><meta data-hydrostats="publish"> <!-- --> <!-- --> <button hidden="" data-testid="header-permalink-button" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="y,Shift+Y"></button><button hidden="" data-hotkey="y,Shift+Y"></button><div><div style="--sticky-pane-height: calc(100vh - (max(184.8000030517578px, 0px))); --spacing: var(--spacing-none);" class="Box-sc-g0xbh4-0 prc-PageLayout-PageLayoutRoot-1zlEO"><div class="Box-sc-g0xbh4-0 prc-PageLayout-PageLayoutWrapper-s2ao4" data-width="full"><div class="Box-sc-g0xbh4-0 prc-PageLayout-PageLayoutContent-jzDMn"><div tabindex="0" class="Box-sc-g0xbh4-0 gISSDQ"><div class="Box-sc-g0xbh4-0 prc-PageLayout-PaneWrapper-nGO0U ReposFileTreePane-module__Pane--wS7IV ReposFileTreePane-module__HidePane--Gj4XZ" style="--offset-header:0px;--spacing-row:var(--spacing-none);--spacing-column:var(--spacing-none)" data-is-hidden="false" data-position="start" data-sticky="true"><div class="Box-sc-g0xbh4-0 prc-PageLayout-HorizontalDivider-CYLp5 prc-PageLayout-PaneHorizontalDivider-4exOb" data-variant="none" data-position="start" style="--spacing-divider:var(--spacing-none);--spacing:var(--spacing-none)"></div><div class="Box-sc-g0xbh4-0 prc-PageLayout-Pane-Vl5LI" data-resizable="true" style="--spacing:var(--spacing-none);--pane-min-width:256px;--pane-max-width:calc(100vw - var(--pane-max-width-diff));--pane-width-size:var(--pane-width-large);--pane-width:320px"><div><div id="repos-file-tree" class="Box-sc-g0xbh4-0 bNhwaa"><div class="Box-sc-g0xbh4-0 hNNCwk"><div class="Box-sc-g0xbh4-0 jfIeyl"><h2 class="Box-sc-g0xbh4-0 XosP prc-Heading-Heading-6CmGO"><span role="tooltip" aria-label="Collapse file tree" id="expand-button-file-tree-button" class="Tooltip__TooltipBase-sc-17tf59c-0 hWlpPn tooltipped-se"><button data-component="IconButton" type="button" data-testid="collapse-file-tree-button" aria-expanded="true" aria-controls="repos-file-tree" data-hotkey="Control+b" data-analytics-opt-out="true" class="prc-Button-ButtonBase-c50BI position-relative ExpandFileTreeButton-module__expandButton--gL4is fgColor-muted prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="invisible" aria-describedby=":r12:-loading-announcement" aria-labelledby="expand-button-file-tree-button"><svg aria-hidden="true" focusable="false" class="octicon octicon-sidebar-expand" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="m4.177 7.823 2.396-2.396A.25.25 0 0 1 7 5.604v4.792a.25.25 0 0 1-.427.177L4.177 8.177a.25.25 0 0 1 0-.354Z"></path><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25H9.5v-13Zm12.5 13a.25.25 0 0 0 .25-.25V1.75a.25.25 0 0 0-.25-.25H11v13Z"></path></svg></button></span><button hidden="" data-testid="" data-hotkey="Control+b" data-hotkey-scope="read-only-cursor-text-area"></button></h2><h2 class="Box-sc-g0xbh4-0 kOkWgo prc-Heading-Heading-6CmGO">Files</h2></div><div class="Box-sc-g0xbh4-0 lhbroM"><div class="Box-sc-g0xbh4-0 khzwtX"><button type="button" aria-haspopup="true" aria-expanded="false" tabindex="0" data-hotkey="w" aria-label="main branch" data-testid="anchor-button" class="Box-sc-g0xbh4-0 gMOVLe prc-Button-ButtonBase-c50BI react-repos-tree-pane-ref-selector width-full ref-selector-class" data-loading="false" data-size="medium" data-variant="default" aria-describedby="branch-picker-repos-header-ref-selector-loading-announcement" id="branch-picker-repos-header-ref-selector"><span data-component="buttonContent" class="Box-sc-g0xbh4-0 gUkoLg prc-Button-ButtonContent-HKbr-"><span data-component="text" class="prc-Button-Label-pTQ3x"><div class="Box-sc-g0xbh4-0 bZBlpz"><div class="Box-sc-g0xbh4-0 lhTYNA"><svg aria-hidden="true" focusable="false" class="octicon octicon-git-branch" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M9.5 3.25a2.25 2.25 0 1 1 3 2.122V6A2.5 2.5 0 0 1 10 8.5H6a1 1 0 0 0-1 1v1.128a2.251 2.251 0 1 1-1.5 0V5.372a2.25 2.25 0 1 1 1.5 0v1.836A2.493 2.493 0 0 1 6 7h4a1 1 0 0 0 1-1v-.628A2.25 2.25 0 0 1 9.5 3.25Zm-6 0a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Zm8.25-.75a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5ZM4.25 12a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Z"></path></svg></div><div class="Box-sc-g0xbh4-0 ffLUq ref-selector-button-text-container"><span class="Box-sc-g0xbh4-0 bmcJak prc-Text-Text-0ima0">&nbsp;main</span></div></div></span><span data-component="trailingVisual" class="prc-Button-Visual-2epfX prc-Button-VisualWrap-Db-eB"><svg aria-hidden="true" focusable="false" class="octicon octicon-triangle-down" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="m4.427 7.427 3.396 3.396a.25.25 0 0 0 .354 0l3.396-3.396A.25.25 0 0 0 11.396 7H4.604a.25.25 0 0 0-.177.427Z"></path></svg></span></span></button><button hidden="" data-hotkey="w" data-hotkey-scope="read-only-cursor-text-area"></button></div><div class="Box-sc-g0xbh4-0 eTeVqd"><button data-component="IconButton" type="button" aria-label="Search this repository" data-hotkey="/" class="Box-sc-g0xbh4-0 dHRvks prc-Button-ButtonBase-c50BI prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" aria-describedby=":r15:-loading-announcement"><svg aria-hidden="true" focusable="false" class="octicon octicon-search" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"></path></svg></button><button hidden="" data-testid="" data-hotkey="/" data-hotkey-scope="read-only-cursor-text-area"></button></div></div></div><div class="Box-sc-g0xbh4-0 qkmJR"><span class="Box-sc-g0xbh4-0 vcvyP TextInput-wrapper prc-components-TextInputWrapper-i1ofR prc-components-TextInputBaseWrapper-ueK9q" data-leading-visual="true" data-trailing-visual="true" aria-busy="false"><span class="TextInput-icon" id=":r16:" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-search" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"></path></svg></span><input type="text" aria-label="Go to file" role="combobox" aria-controls="file-results-list" aria-expanded="false" aria-haspopup="dialog" autocorrect="off" spellcheck="false" placeholder="Go to file" aria-describedby=":r16: :r17:" data-component="input" class="prc-components-Input-Ic-y8" value=""><span class="TextInput-icon" id=":r17:" aria-hidden="true"></span></span></div><button hidden="" data-testid="" data-hotkey="t,Shift+T" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="t,Shift+T"></button><div class="Box-sc-g0xbh4-0 jbQqON"><div><div class="react-tree-show-tree-items"><div class="Box-sc-g0xbh4-0 cOxzdh" data-testid="repos-file-tree-container"><nav aria-label="File Tree Navigation"><span role="status" aria-live="polite" aria-atomic="true" class="_VisuallyHidden__VisuallyHidden-sc-11jhm7a-0 brGdpi"></span><ul role="tree" aria-label="Files" data-truncate-text="true" class="prc-TreeView-TreeViewRootUlStyles-eZtxW"><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="metrics-item" role="treeitem" aria-labelledby=":r0:" aria-describedby=":r1:" aria-level="1" aria-expanded="false" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div class="PRIVATE_TreeView-item-toggle PRIVATE_TreeView-item-toggle--hover PRIVATE_TreeView-item-toggle--end prc-TreeView-TreeViewItemToggle-gWUdE prc-TreeView-TreeViewItemToggleHover-nEgP- prc-TreeView-TreeViewItemToggleEnd-t-AEB"><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right" viewBox="0 0 12 12" width="12" height="12" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M4.7 10c-.2 0-.4-.1-.5-.2-.3-.3-.3-.8 0-1.1L6.9 6 4.2 3.3c-.3-.3-.3-.8 0-1.1.3-.3.8-.3 1.1 0l3.3 3.2c.3.3.3.8 0 1.1L5.3 9.7c-.2.2-.4.3-.6.3Z"></path></svg></div><div id=":r0:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":r1:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><div class="PRIVATE_TreeView-directory-icon prc-TreeView-TreeViewDirectoryIcon-PHbeP"><svg aria-hidden="true" focusable="false" class="octicon octicon-file-directory-fill" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M1.75 1A1.75 1.75 0 0 0 0 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0 0 16 13.25v-8.5A1.75 1.75 0 0 0 14.25 3H7.5a.25.25 0 0 1-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75Z"></path></svg></div></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>metrics</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="models-item" role="treeitem" aria-labelledby=":r3:" aria-describedby=":r4:" aria-level="1" aria-expanded="false" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div class="PRIVATE_TreeView-item-toggle PRIVATE_TreeView-item-toggle--hover PRIVATE_TreeView-item-toggle--end prc-TreeView-TreeViewItemToggle-gWUdE prc-TreeView-TreeViewItemToggleHover-nEgP- prc-TreeView-TreeViewItemToggleEnd-t-AEB"><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right" viewBox="0 0 12 12" width="12" height="12" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M4.7 10c-.2 0-.4-.1-.5-.2-.3-.3-.3-.8 0-1.1L6.9 6 4.2 3.3c-.3-.3-.3-.8 0-1.1.3-.3.8-.3 1.1 0l3.3 3.2c.3.3.3.8 0 1.1L5.3 9.7c-.2.2-.4.3-.6.3Z"></path></svg></div><div id=":r3:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":r4:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><div class="PRIVATE_TreeView-directory-icon prc-TreeView-TreeViewDirectoryIcon-PHbeP"><svg aria-hidden="true" focusable="false" class="octicon octicon-file-directory-fill" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M1.75 1A1.75 1.75 0 0 0 0 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0 0 16 13.25v-8.5A1.75 1.75 0 0 0 14.25 3H7.5a.25.25 0 0 1-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75Z"></path></svg></div></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>models</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="static-item" role="treeitem" aria-labelledby=":r6:" aria-describedby=":r7:" aria-level="1" aria-expanded="false" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div class="PRIVATE_TreeView-item-toggle PRIVATE_TreeView-item-toggle--hover PRIVATE_TreeView-item-toggle--end prc-TreeView-TreeViewItemToggle-gWUdE prc-TreeView-TreeViewItemToggleHover-nEgP- prc-TreeView-TreeViewItemToggleEnd-t-AEB"><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right" viewBox="0 0 12 12" width="12" height="12" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M4.7 10c-.2 0-.4-.1-.5-.2-.3-.3-.3-.8 0-1.1L6.9 6 4.2 3.3c-.3-.3-.3-.8 0-1.1.3-.3.8-.3 1.1 0l3.3 3.2c.3.3.3.8 0 1.1L5.3 9.7c-.2.2-.4.3-.6.3Z"></path></svg></div><div id=":r6:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":r7:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><div class="PRIVATE_TreeView-directory-icon prc-TreeView-TreeViewDirectoryIcon-PHbeP"><svg aria-hidden="true" focusable="false" class="octicon octicon-file-directory-fill" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M1.75 1A1.75 1.75 0 0 0 0 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0 0 16 13.25v-8.5A1.75 1.75 0 0 0 14.25 3H7.5a.25.25 0 0 1-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75Z"></path></svg></div></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>static</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="templates-item" role="treeitem" aria-labelledby=":r9:" aria-describedby=":ra:" aria-level="1" aria-expanded="false" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div class="PRIVATE_TreeView-item-toggle PRIVATE_TreeView-item-toggle--hover PRIVATE_TreeView-item-toggle--end prc-TreeView-TreeViewItemToggle-gWUdE prc-TreeView-TreeViewItemToggleHover-nEgP- prc-TreeView-TreeViewItemToggleEnd-t-AEB"><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right" viewBox="0 0 12 12" width="12" height="12" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M4.7 10c-.2 0-.4-.1-.5-.2-.3-.3-.3-.8 0-1.1L6.9 6 4.2 3.3c-.3-.3-.3-.8 0-1.1.3-.3.8-.3 1.1 0l3.3 3.2c.3.3.3.8 0 1.1L5.3 9.7c-.2.2-.4.3-.6.3Z"></path></svg></div><div id=":r9:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":ra:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><div class="PRIVATE_TreeView-directory-icon prc-TreeView-TreeViewDirectoryIcon-PHbeP"><svg aria-hidden="true" focusable="false" class="octicon octicon-file-directory-fill" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M1.75 1A1.75 1.75 0 0 0 0 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0 0 16 13.25v-8.5A1.75 1.75 0 0 0 14.25 3H7.5a.25.25 0 0 1-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75Z"></path></svg></div></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>templates</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="0" id="App.py-item" role="treeitem" aria-labelledby=":rc:" aria-describedby=":rd:" aria-level="1" aria-current="true" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id=":rc:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":rd:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>App.py</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="LICENSE-item" role="treeitem" aria-labelledby=":rf:" aria-describedby=":rg:" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id=":rf:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":rg:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>LICENSE</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="README.md-item" role="treeitem" aria-labelledby=":ri:" aria-describedby=":rj:" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id=":ri:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":rj:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>README.md</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="Sample_2.mp4-item" role="treeitem" aria-labelledby=":rl:" aria-describedby=":rm:" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id=":rl:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":rm:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>Sample_2.mp4</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="Train_Blink.ipynb-item" role="treeitem" aria-labelledby=":ro:" aria-describedby=":rp:" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id=":ro:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":rp:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>Train_Blink.ipynb</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="Train_Gaze_Updated.ipynb-item" role="treeitem" aria-labelledby=":rr:" aria-describedby=":rs:" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id=":rr:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":rs:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>Train_Gaze_Updated.ipynb</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-ShJr0" tabindex="-1" id="requirements.txt-item" role="treeitem" aria-labelledby=":ru:" aria-describedby=":rv:" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer--2Rkn" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id=":ru:" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-f0r0b"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-4-mPv" aria-hidden="true" id=":rv:"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-dRlGq" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-smZM-"><span>requirements.txt</span></span></div></div></li></ul></nav></div></div></div></div></div></div></div><div class="Box-sc-g0xbh4-0 prc-PageLayout-VerticalDivider-4A4Qm prc-PageLayout-PaneVerticalDivider-1c9vy" data-variant="line" data-position="start" style="--spacing:var(--spacing-none)"><div role="slider" aria-label="Draggable pane splitter" aria-valuemin="256" aria-valuemax="587" aria-valuenow="0" aria-valuetext="Pane width 0 pixels" tabindex="0" class="Box-sc-g0xbh4-0 bHLmSv"></div></div></div></div><div class="Box-sc-g0xbh4-0 prc-PageLayout-ContentWrapper-b-QRo CodeView-module__SplitPageLayout_Content--qxR1C" data-is-hidden="false"><div class="Box-sc-g0xbh4-0"></div><div class="Box-sc-g0xbh4-0 prc-PageLayout-Content--F7-I" data-width="full" style="--spacing:var(--spacing-none)"><div data-selector="repos-split-pane-content" tabindex="0" class="Box-sc-g0xbh4-0 leYMvG"><div class="Box-sc-g0xbh4-0 KMPzq"><div class="Box-sc-g0xbh4-0 hfKjHv container"><div class="px-3 pt-3 pb-0" id="StickyHeader"><div class="Box-sc-g0xbh4-0 gZWyZE"><div class="Box-sc-g0xbh4-0 dwYKDk"><div class="Box-sc-g0xbh4-0 iDtIiT"><div class="Box-sc-g0xbh4-0 cEytCf"><nav data-testid="breadcrumbs" aria-labelledby="repos-header-breadcrumb--wide-heading" id="repos-header-breadcrumb--wide" class="Box-sc-g0xbh4-0 fzFXnm"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone--vW4Cq prc-Heading-Heading-6CmGO" data-testid="screen-reader-heading" id="repos-header-breadcrumb--wide-heading">Breadcrumbs</h2><ol class="Box-sc-g0xbh4-0 iMnkmv"><li class="Box-sc-g0xbh4-0 ghzDag"><a class="Box-sc-g0xbh4-0 kHuKdh prc-Link-Link-85e08" sx="[object Object]" data-testid="breadcrumbs-repo-link" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/tree/main">Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection</a></li></ol></nav><div data-testid="breadcrumbs-filename" class="Box-sc-g0xbh4-0 ghzDag"><span class="Box-sc-g0xbh4-0 hzJBof prc-Text-Text-0ima0" aria-hidden="true">/</span><h1 class="Box-sc-g0xbh4-0 jGhzSQ prc-Heading-Heading-6CmGO" tabindex="-1" id="file-name-id-wide">App.py</h1></div><button data-component="IconButton" type="button" class="prc-Button-ButtonBase-c50BI ml-2 prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="small" data-variant="invisible" aria-describedby=":r1c:-loading-announcement" aria-labelledby=":r1a:"><svg aria-hidden="true" focusable="false" class="octicon octicon-copy" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></button><span class="CopyToClipboardButton-module__tooltip--Dq1IB prc-TooltipV2-Tooltip-cYMVY" data-direction="nw" aria-label="Copy path" aria-hidden="true" id=":r1a:" popover="auto">Copy path</span></div></div><div class="react-code-view-header-element--wide"><div class="Box-sc-g0xbh4-0 faNtbn"><div class="d-flex gap-2"> <button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="l,Shift+L"></button><button hidden="" data-hotkey="l,Shift+L"></button><button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="Mod+Alt+g"></button><button hidden="" data-hotkey="Mod+Alt+g"></button><button type="button" class="Box-sc-g0xbh4-0 dwNhzn prc-Button-ButtonBase-c50BI" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" aria-describedby=":R5a6d9lab:-loading-announcement" data-hotkey="b,Shift+B,Control+/ Control+b"><span data-component="buttonContent" class="Box-sc-g0xbh4-0 gUkoLg prc-Button-ButtonContent-HKbr-"><span data-component="text" class="prc-Button-Label-pTQ3x">Blame</span></span></button><button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="b,Shift+B,Control+/ Control+b"></button><button data-component="IconButton" type="button" aria-label="More file actions" title="More file actions" data-testid="more-file-actions-button-nav-menu-wide" aria-haspopup="true" aria-expanded="false" tabindex="0" class="Box-sc-g0xbh4-0 fGwBZA prc-Button-ButtonBase-c50BI js-blob-dropdown-click prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" aria-describedby=":R2a6d9lab:-loading-announcement" id=":R2a6d9lab:"><svg aria-hidden="true" focusable="false" class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M8 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Zm13 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path></svg></button> </div></div></div><div class="react-code-view-header-element--narrow"><div class="Box-sc-g0xbh4-0 faNtbn"><div class="d-flex gap-2"> <button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="l,Shift+L"></button><button hidden="" data-hotkey="l,Shift+L"></button><button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="Mod+Alt+g"></button><button hidden="" data-hotkey="Mod+Alt+g"></button><button type="button" class="Box-sc-g0xbh4-0 dwNhzn prc-Button-ButtonBase-c50BI" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" aria-describedby=":R5a7d9lab:-loading-announcement" data-hotkey="b,Shift+B,Control+/ Control+b"><span data-component="buttonContent" class="Box-sc-g0xbh4-0 gUkoLg prc-Button-ButtonContent-HKbr-"><span data-component="text" class="prc-Button-Label-pTQ3x">Blame</span></span></button><button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="b,Shift+B,Control+/ Control+b"></button><button data-component="IconButton" type="button" aria-label="More file actions" title="More file actions" data-testid="more-file-actions-button-nav-menu-narrow" aria-haspopup="true" aria-expanded="false" tabindex="0" class="Box-sc-g0xbh4-0 fGwBZA prc-Button-ButtonBase-c50BI js-blob-dropdown-click prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" aria-describedby=":R2a7d9lab:-loading-announcement" id=":R2a7d9lab:"><svg aria-hidden="true" focusable="false" class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M8 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Zm13 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path></svg></button> </div></div></div></div></div></div></div></div><div class="Box-sc-g0xbh4-0 dJxjrT react-code-view-bottom-padding"> <div class="Box-sc-g0xbh4-0 eFxKDQ"></div> <!-- --> <!-- --> </div><div class="Box-sc-g0xbh4-0 dJxjrT"> <!-- --> <!-- --> <button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="r,Shift+R"></button><button hidden="" data-hotkey="r,Shift+R"></button><div class="d-flex flex-column border rounded-2 mb-3 pl-1"><div class="Box-sc-g0xbh4-0 dzCJzi"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone--vW4Cq prc-Heading-Heading-6CmGO" data-testid="screen-reader-heading">Latest commit</h2><div data-testid="latest-commit" class="Box-sc-g0xbh4-0 ePWWCk"><div class="Box-sc-g0xbh4-0 dpBUfI"><div data-testid="author-avatar" class="Box-sc-g0xbh4-0 hKWjvQ"><a class="prc-Link-Link-85e08" href="https://github.com/Rafael-ZP" data-testid="avatar-icon-link" data-hovercard-url="/users/Rafael-ZP/hovercard"><img data-component="Avatar" class="Box-sc-g0xbh4-0 cvdqJW prc-Avatar-Avatar-ZRS-m" alt="Rafael-ZP" width="20" height="20" src="./App_files/104310982" data-testid="github-avatar" aria-label="Rafael-ZP" style="--avatarSize-regular: 20px;"></a><a class="Box-sc-g0xbh4-0 dkaFxu prc-Link-Link-85e08" data-muted="true" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/commits?author=Rafael-ZP" aria-label="commits by Rafael-ZP" data-hovercard-url="/users/Rafael-ZP/hovercard">Rafael-ZP</a></div><span class=""></span></div><div class="Box-sc-g0xbh4-0 erEOeb d-none d-sm-flex"><div class="Truncate flex-items-center f5"><span class="Truncate-text prc-Text-Text-0ima0" data-testid="latest-commit-html"><a href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/commit/76b21dd03dc92bda3da713127dce9642fa208801" class="Link--secondary" data-pjax="true" data-hovercard-url="/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/commit/76b21dd03dc92bda3da713127dce9642fa208801/hovercard">Add files via upload</a></span></div></div><span class="d-flex d-sm-none fgColor-muted f6"><relative-time class="sc-aXZVg" tense="past" datetime="2025-04-10T09:49:27.000Z" title="Apr 10, 2025, 3:19 PM GMT+5:30"><template shadowrootmode="open">4 hours ago</template>Apr 10, 2025</relative-time></span></div><div class="d-flex flex-shrink-0 gap-2"><div data-testid="latest-commit-details" class="d-none d-sm-flex flex-items-center"><span class="d-flex flex-nowrap fgColor-muted f6"><a class="Link--secondary prc-Link-Link-85e08" aria-label="Commit 76b21dd" data-hovercard-url="/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/commit/76b21dd03dc92bda3da713127dce9642fa208801/hovercard" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/commit/76b21dd03dc92bda3da713127dce9642fa208801">76b21dd</a>&nbsp;·&nbsp;<relative-time class="sc-aXZVg" tense="past" datetime="2025-04-10T09:49:27.000Z" title="Apr 10, 2025, 3:19 PM GMT+5:30"><template shadowrootmode="open">4 hours ago</template>Apr 10, 2025</relative-time></span></div><div class="d-flex gap-2"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone--vW4Cq prc-Heading-Heading-6CmGO" data-testid="screen-reader-heading">History</h2><a href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/commits/main/App.py" class="prc-Button-ButtonBase-c50BI d-none d-lg-flex LinkButton-module__code-view-link-button--xvCGA flex-items-center fgColor-default" data-loading="false" data-size="small" data-variant="invisible" aria-describedby=":R5dlal9lab:-loading-announcement"><span data-component="buttonContent" data-align="center" class="prc-Button-ButtonContent-HKbr-"><span data-component="leadingVisual" class="prc-Button-Visual-2epfX prc-Button-VisualWrap-Db-eB"><svg aria-hidden="true" focusable="false" class="octicon octicon-history" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="m.427 1.927 1.215 1.215a8.002 8.002 0 1 1-1.6 5.685.75.75 0 1 1 1.493-.154 6.5 6.5 0 1 0 1.18-4.458l1.358 1.358A.25.25 0 0 1 3.896 6H.25A.25.25 0 0 1 0 5.75V2.104a.25.25 0 0 1 .427-.177ZM7.75 4a.75.75 0 0 1 .75.75v2.992l2.028.812a.75.75 0 0 1-.557 1.392l-2.5-1A.751.751 0 0 1 7 8.25v-3.5A.75.75 0 0 1 7.75 4Z"></path></svg></span><span data-component="text" class="prc-Button-Label-pTQ3x"><span class="fgColor-default">History</span></span></span></a><div class="d-sm-none"><button data-component="IconButton" type="button" aria-label="Open commit details" aria-pressed="false" aria-expanded="false" data-testid="latest-commit-details-toggle" class="Box-sc-g0xbh4-0 hdOVEE prc-Button-ButtonBase-c50BI prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="small" data-variant="invisible" aria-describedby=":r1k:-loading-announcement"><svg aria-hidden="true" focusable="false" class="octicon octicon-ellipsis" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M0 5.75C0 4.784.784 4 1.75 4h12.5c.966 0 1.75.784 1.75 1.75v4.5A1.75 1.75 0 0 1 14.25 12H1.75A1.75 1.75 0 0 1 0 10.25ZM12 7a1 1 0 1 0 0 2 1 1 0 0 0 0-2ZM7 8a1 1 0 1 0 2 0 1 1 0 0 0-2 0ZM4 7a1 1 0 1 0 0 2 1 1 0 0 0 0-2Z"></path></svg></button></div><div class="d-flex d-lg-none"><span role="tooltip" aria-label="History" id="history-icon-button-tooltip" class="Tooltip__TooltipBase-sc-17tf59c-0 hWlpPn tooltipped-n"><a href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/commits/main/App.py" class="prc-Button-ButtonBase-c50BI LinkButton-module__code-view-link-button--xvCGA flex-items-center fgColor-default" data-loading="false" data-size="small" data-variant="invisible" aria-describedby=":Rpdlal9lab:-loading-announcement history-icon-button-tooltip"><span data-component="buttonContent" data-align="center" class="prc-Button-ButtonContent-HKbr-"><span data-component="leadingVisual" class="prc-Button-Visual-2epfX prc-Button-VisualWrap-Db-eB"><svg aria-hidden="true" focusable="false" class="octicon octicon-history" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="m.427 1.927 1.215 1.215a8.002 8.002 0 1 1-1.6 5.685.75.75 0 1 1 1.493-.154 6.5 6.5 0 1 0 1.18-4.458l1.358 1.358A.25.25 0 0 1 3.896 6H.25A.25.25 0 0 1 0 5.75V2.104a.25.25 0 0 1 .427-.177ZM7.75 4a.75.75 0 0 1 .75.75v2.992l2.028.812a.75.75 0 0 1-.557 1.392l-2.5-1A.751.751 0 0 1 7 8.25v-3.5A.75.75 0 0 1 7.75 4Z"></path></svg></span></span></a></span></div></div></div></div></div><div class="Box-sc-g0xbh4-0 ldRxiI"><div class="Box-sc-g0xbh4-0 fVkfyA container"><div class="Box-sc-g0xbh4-0 gNAmSV react-code-size-details-banner"><div class="Box-sc-g0xbh4-0 jNEwzY react-code-size-details-banner"><div class="Box-sc-g0xbh4-0 bsDwxw text-mono"><div title="11.1 KB" data-testid="blob-size" class="Truncate__StyledTruncate-sc-23o1d2-0 eAtkQz"><span>239 lines (206 loc) · 11.1 KB</span></div></div></div></div><div class="Box-sc-g0xbh4-0 jdLMhu react-blob-view-header-sticky" id="repos-sticky-header"><div class="Box-sc-g0xbh4-0 tOISc"><div class="react-blob-sticky-header"><div class="Box-sc-g0xbh4-0 hqwSEx"><div class="Box-sc-g0xbh4-0 lzKZY"><div class="Box-sc-g0xbh4-0 fHind"><nav data-testid="breadcrumbs" aria-labelledby="sticky-breadcrumb-heading" id="sticky-breadcrumb" class="Box-sc-g0xbh4-0 fzFXnm"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone--vW4Cq prc-Heading-Heading-6CmGO" data-testid="screen-reader-heading" id="sticky-breadcrumb-heading">Breadcrumbs</h2><ol class="Box-sc-g0xbh4-0 iMnkmv"><li class="Box-sc-g0xbh4-0 ghzDag"><a class="Box-sc-g0xbh4-0 kHuKdh prc-Link-Link-85e08" sx="[object Object]" data-testid="breadcrumbs-repo-link" href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/tree/main">Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection</a></li></ol></nav><div data-testid="breadcrumbs-filename" class="Box-sc-g0xbh4-0 ghzDag"><span class="Box-sc-g0xbh4-0 oDtgN prc-Text-Text-0ima0" aria-hidden="true">/</span><h1 class="Box-sc-g0xbh4-0 dnZoUW prc-Heading-Heading-6CmGO" tabindex="-1" id="sticky-file-name-id">App.py</h1></div></div><button style="--button-color:fg.default" type="button" class="Box-sc-g0xbh4-0 jRZWlf prc-Button-ButtonBase-c50BI" data-loading="false" data-size="small" data-variant="invisible" aria-describedby=":Riptal9lab:-loading-announcement"><span data-component="buttonContent" class="Box-sc-g0xbh4-0 gUkoLg prc-Button-ButtonContent-HKbr-"><span data-component="leadingVisual" class="prc-Button-Visual-2epfX prc-Button-VisualWrap-Db-eB"><svg aria-hidden="true" focusable="false" class="octicon octicon-arrow-up" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.47 7.78a.75.75 0 0 1 0-1.06l4.25-4.25a.75.75 0 0 1 1.06 0l4.25 4.25a.751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018L9 4.81v7.44a.75.75 0 0 1-1.5 0V4.81L4.53 7.78a.75.75 0 0 1-1.06 0Z"></path></svg></span><span data-component="text" class="prc-Button-Label-pTQ3x">Top</span></span></button></div></div></div><div class="Box-sc-g0xbh4-0 kTvpNk"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone--vW4Cq prc-Heading-Heading-6CmGO" data-testid="screen-reader-heading">File metadata and controls</h2><div class="Box-sc-g0xbh4-0 iNMjfP"><ul aria-label="File view" class="Box-sc-g0xbh4-0 gtTaSn prc-SegmentedControl-SegmentedControl-e7570" data-size="small"><li class="Box-sc-g0xbh4-0 dXYHoy prc-SegmentedControl-Item-7Aq6h" data-selected="true"><button aria-current="true" class="prc-SegmentedControl-Button-ojWXD" type="button" data-hotkey="Control+/ Control+c"><span class="prc-SegmentedControl-Content-gnQ4n"><div class="Box-sc-g0xbh4-0 prc-SegmentedControl-Text-c5gSh" data-text="Code">Code</div></span></button></li><li class="Box-sc-g0xbh4-0 jBWIdY prc-SegmentedControl-Item-7Aq6h"><button aria-current="false" class="prc-SegmentedControl-Button-ojWXD" type="button" data-hotkey="b,Shift+B,Control+/ Control+b"><span class="prc-SegmentedControl-Content-gnQ4n"><div class="Box-sc-g0xbh4-0 prc-SegmentedControl-Text-c5gSh" data-text="Blame">Blame</div></span></button></li></ul><button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="Control+/ Control+c"></button><button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="b,Shift+B,Control+/ Control+b"></button><div class="Box-sc-g0xbh4-0 jNEwzY react-code-size-details-in-header"><div class="Box-sc-g0xbh4-0 bsDwxw text-mono"><div title="11.1 KB" data-testid="blob-size" class="Truncate__StyledTruncate-sc-23o1d2-0 eAtkQz"><span>239 lines (206 loc) · 11.1 KB</span></div></div></div></div><div class="Box-sc-g0xbh4-0 kcLCKF"><button hidden="" data-testid="" data-hotkey="Control+Shift+&gt;" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="Control+Shift+&gt;"></button><button hidden="" data-testid="" data-hotkey="Control+Shift+&lt;" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="Control+Shift+&lt;"></button><div class="Box-sc-g0xbh4-0 kVWtTz react-blob-header-edit-and-raw-actions"><div class="Box-sc-g0xbh4-0 prc-ButtonGroup-ButtonGroup-vcMeG"><div><a href="https://github.com/Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/raw/refs/heads/main/App.py" data-testid="raw-button" class="Box-sc-g0xbh4-0 gWqxTd prc-Button-ButtonBase-c50BI" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default" aria-describedby=":R5csptal9lab:-loading-announcement" data-hotkey="Control+/ Control+r"><span data-component="buttonContent" class="Box-sc-g0xbh4-0 gUkoLg prc-Button-ButtonContent-HKbr-"><span data-component="text" class="prc-Button-Label-pTQ3x">Raw</span></span></a></div><div><button data-component="IconButton" type="button" aria-label="Copy raw content" data-testid="copy-raw-button" class="prc-Button-ButtonBase-c50BI prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default" aria-describedby=":Rpcsptal9lab:-loading-announcement" data-hotkey="Control+Shift+C"><svg aria-hidden="true" focusable="false" class="octicon octicon-copy" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></button></div><div><span role="tooltip" aria-label="Download raw file" id=":Rdcsptal9lab:" class="Tooltip__TooltipBase-sc-17tf59c-0 hWlpPn tooltipped-n"><button data-component="IconButton" type="button" aria-label="Download raw content" data-testid="download-raw-button" class="Box-sc-g0xbh4-0 ivobqY prc-Button-ButtonBase-c50BI prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default" aria-describedby=":Rtcsptal9lab:-loading-announcement" data-hotkey="Control+Shift+S"><svg aria-hidden="true" focusable="false" class="octicon octicon-download" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-2.5a.75.75 0 0 1 1.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 13.25 14Z"></path><path d="M7.25 7.689V2a.75.75 0 0 1 1.5 0v5.689l1.97-1.969a.749.749 0 1 1 1.06 1.06l-3.25 3.25a.749.749 0 0 1-1.06 0L4.22 6.78a.749.749 0 1 1 1.06-1.06l1.97 1.969Z"></path></svg></button></span></div></div><button hidden="" data-testid="raw-button-shortcut" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="Control+/ Control+r"></button><button hidden="" data-testid="copy-raw-button-shortcut" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="Control+Shift+C"></button><button hidden="" data-testid="download-raw-button-shortcut" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="Control+Shift+S"></button><div class="Box-sc-g0xbh4-0 prc-ButtonGroup-ButtonGroup-vcMeG"><div><span role="tooltip" aria-label="You must be signed in to make or propose changes" id=":r1d:" class="Tooltip__TooltipBase-sc-17tf59c-0 hWlpPn tooltipped-nw"><button aria-disabled="true" data-component="IconButton" type="button" aria-label="Edit file" class="Box-sc-g0xbh4-0 hDtCBD prc-Button-ButtonBase-c50BI prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default" aria-describedby=":r1e:-loading-announcement"><svg aria-hidden="true" focusable="false" class="octicon octicon-pencil" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M11.013 1.427a1.75 1.75 0 0 1 2.474 0l1.086 1.086a1.75 1.75 0 0 1 0 2.474l-8.61 8.61c-.21.21-.47.364-.756.445l-3.251.93a.75.75 0 0 1-.927-.928l.929-3.25c.081-.286.235-.547.445-.758l8.61-8.61Zm.176 4.823L9.75 4.81l-6.286 6.287a.253.253 0 0 0-.064.108l-.558 1.953 1.953-.558a.253.253 0 0 0 .108-.064Zm1.238-3.763a.25.25 0 0 0-.354 0L10.811 3.75l1.439 1.44 1.263-1.263a.25.25 0 0 0 0-.354Z"></path></svg></button></span></div><div><button data-component="IconButton" type="button" aria-label="More edit options" data-testid="more-edit-button" aria-haspopup="true" aria-expanded="false" tabindex="0" class="prc-Button-ButtonBase-c50BI prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default" aria-describedby=":r1f:-loading-announcement" id=":r1f:"><svg aria-hidden="true" focusable="false" class="octicon octicon-triangle-down" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="m4.427 7.427 3.396 3.396a.25.25 0 0 0 .354 0l3.396-3.396A.25.25 0 0 0 11.396 7H4.604a.25.25 0 0 0-.177.427Z"></path></svg></button></div></div></div><span role="tooltip" aria-label="Open symbols panel" id=":R5sptal9lab:" class="Tooltip__TooltipBase-sc-17tf59c-0 hWlpPn tooltipped-nw"><button data-component="IconButton" type="button" aria-label="Symbols" aria-pressed="false" aria-expanded="false" aria-controls="symbols-pane" data-testid="symbols-button" class="Box-sc-g0xbh4-0 heuRGy prc-Button-ButtonBase-c50BI prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="small" data-variant="invisible" aria-describedby="symbols-button-loading-announcement" id="symbols-button" data-hotkey="Control+i"><svg aria-hidden="true" focusable="false" class="octicon octicon-code-square" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h12.5a.25.25 0 0 0 .25-.25V1.75a.25.25 0 0 0-.25-.25Zm7.47 3.97a.75.75 0 0 1 1.06 0l2 2a.75.75 0 0 1 0 1.06l-2 2a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734L10.69 8 9.22 6.53a.75.75 0 0 1 0-1.06ZM6.78 6.53 5.31 8l1.47 1.47a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215l-2-2a.75.75 0 0 1 0-1.06l2-2a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"></path></svg></button></span><div class="react-blob-header-edit-and-raw-actions-combined"><button data-component="IconButton" type="button" aria-label="Edit and raw actions" title="More file actions" data-testid="more-file-actions-button" aria-haspopup="true" aria-expanded="false" tabindex="0" class="Box-sc-g0xbh4-0 ffkqe prc-Button-ButtonBase-c50BI js-blob-dropdown-click prc-Button-IconButton-szpyj" data-loading="false" data-no-visuals="true" data-size="small" data-variant="invisible" aria-describedby=":Rnsptal9lab:-loading-announcement" id=":Rnsptal9lab:"><svg aria-hidden="true" focusable="false" class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M8 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Zm13 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path></svg></button></div></div></div></div><div></div></div><div class="Box-sc-g0xbh4-0 hGyMdv"><section aria-labelledby="file-name-id-wide file-name-id-mobile" class="Box-sc-g0xbh4-0 dceWRL"><div class="Box-sc-g0xbh4-0 dGXHv"><div id="highlighted-line-menu-positioner" class="position-relative"><div id="copilot-button-positioner" class="Box-sc-g0xbh4-0 bpDFns"><div class="Box-sc-g0xbh4-0 iJOeCH"><div class="Box-sc-g0xbh4-0 eJSJhL"><div class="Box-sc-g0xbh4-0 jQvQqQ"><div aria-hidden="true" data-testid="navigation-cursor" class="Box-sc-g0xbh4-0 code-navigation-cursor" style="top: 0px; left: 92px;"> </div><button hidden="" data-testid="NavigationCursorEnter" data-hotkey="Control+Enter" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="NavigationCursorSetHighlightedLine" data-hotkey="Shift+J" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="NavigationCursorSetHighlightAndExpandMenu" data-hotkey="Alt+Shift+C,Alt+Shift+Ç" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="NavigationCursorPageDown" data-hotkey="PageDown" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="NavigationCursorPageUp" data-hotkey="PageUp" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="" data-hotkey="/" data-hotkey-scope="read-only-cursor-text-area"></button></div></div><textarea id="read-only-cursor-text-area" data-testid="read-only-cursor-text-area" aria-label="file content" aria-readonly="true" inputmode="none" tabindex="0" aria-multiline="true" aria-haspopup="false" data-gramm="false" data-gramm_editor="false" data-enable-grammarly="false" spellcheck="false" autocorrect="off" autocapitalize="off" autocomplete="off" data-ms-editor="false" class="react-blob-textarea react-blob-print-hide" style="resize: none; margin-top: -2px; padding-left: 92px; padding-right: 70px; width: 100%; background-color: unset; box-sizing: border-box; color: transparent; position: absolute; border: none; tab-size: 8; outline: none; overflow: auto hidden; height: 4800px; font-size: 12px; line-height: 20px; overflow-wrap: normal; overscroll-behavior-x: none; white-space: pre; z-index: 1;">import cv2
import dlib
import numpy as np
import time
from tensorflow.keras.models import load_model
from joblib import load as joblib_load

# -----------------------------
# Load Haar cascades and Dlib model
# -----------------------------
face_cascade = cv2.CascadeClassifier("/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/haarcascade_eye.xml")
predictor_path = "/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# -----------------------------
# Load your two trained ML models
# -----------------------------
# 1. Fine tuned MobileNetV2 for looking / not looking
#    (expects full face color image resized to 224x224)
looking_model_path = "/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/fine_tuned_mobilenetv2.h5"
looking_model = load_model(looking_model_path)

# 2. Blink SVM for open/closed eye classification
#    (expects a flattened, normalized grayscale image of size 64x64, i.e. 4096 features)
blink_model_path = "/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/blink_svm.pkl"
blink_model = joblib_load(blink_model_path)

# -----------------------------
# Global parameters for attendance logic and fonts
# -----------------------------
ATTENDANCE_DURATION = 10       # seconds required to mark attendance (must be "good" for 10 sec)
ATTENDANCE_MESSAGE_DURATION = 2  # seconds to display the attendance message
MAX_CLOSED_FRAMES = 10         # if eyes are closed for &gt; this many consecutive frames, reset the timer
EYE_INPUT_SIZE = (64, 64)      # expected input size for the blink SVM (64x64 = 4096 features)

# Font scales for on-screen text
FONT_SCALE = 1.2
ATTENDANCE_FONT_SCALE = 2.0
FONT_THICKNESS = 2

def detect_face_eyes(video_source=0):
    """
    Detects face, eyes, and landmarks; then runs the looking model and blink model.
    - The entire face region (in color) is passed to the looking model.
    - The eye regions (cropped from the grayscale image, expanded upward to include eyebrows)
      are resized to 64x64 and passed to the blink SVM.
      
    Prediction logic (inverted if necessary):
      * For the looking model, a probability &lt;= 0.5 indicates "looking."
      * For the blink model, a prediction of 1 indicates "closed" and 0 indicates "open."
      
    If a face is detected, the system checks if the subject is looking. If yes,
    it then checks the eyes. When both eyes are predicted as open (blink SVM returns 0)
    for a continuous period (ATTENDANCE_DURATION seconds) with only brief closures allowed,
    attendance is marked.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # Variables for attendance logic
    good_start_time = None       # Time when "good" criteria first started
    attendance_marked = False    # Whether attendance has been marked already
    attendance_marked_time = None  # Time at which attendance was marked (for display)
    closed_eye_consecutive = 0   # Count of consecutive frames with eyes closed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------------
        # 1. Face detection using Haar cascade
        # -----------------------------
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) &gt; 0:
            # For simplicity, select the largest detected face.
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), FONT_THICKNESS)

            # -----------------------------
            # 2. Looking detection using the entire face crop
            # -----------------------------
            face_roi = frame[y:y + h, x:x + w]
            try:
                face_roi_resized = cv2.resize(face_roi, (224, 224))
            except Exception as e:
                face_roi_resized = face_roi
            face_roi_norm = face_roi_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_roi_norm, axis=0)
            looking_pred = looking_model.predict(face_input)
            # Inverted logic: probability &lt;= 0.5 means "looking"
            if looking_pred[0][0] &lt;= 0.5:
                looking = True
                looking_text = "Looking: Yes"
            else:
                looking = False
                looking_text = "Looking: No"
            cv2.putText(frame, looking_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 0), FONT_THICKNESS)

            # -----------------------------
            # 3. Eye detection using Haar cascade within the face ROI
            # -----------------------------
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), FONT_THICKNESS)

            # -----------------------------
            # 4. Get accurate eye coordinates using dlib’s landmarks
            # -----------------------------
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, dlib_rect)

            # Extract landmarks for left (36-41) and right (42-47) eyes.
            left_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])

            for (ex_pt, ey_pt) in left_eye_pts:
                cv2.circle(frame, (ex_pt, ey_pt), 2, (0, 0, 255), -1)
            for (ex_pt, ey_pt) in right_eye_pts:
                cv2.circle(frame, (ex_pt, ey_pt), 2, (0, 0, 255), -1)

            # Compute bounding boxes for each eye.
            lx, ly, lw, lh = cv2.boundingRect(left_eye_pts)
            rx, ry, rw, rh = cv2.boundingRect(right_eye_pts)

            # Expand upward to include eyebrows (approx. 30% of the eye height)
            eyebrow_offset_left = int(0.3 * lh)
            eyebrow_offset_right = int(0.3 * rh)
            lx_new, ly_new = lx, max(ly - eyebrow_offset_left, 0)
            lw_new, lh_new = lw, lh + eyebrow_offset_left
            rx_new, ry_new = rx, max(ry - eyebrow_offset_right, 0)
            rw_new, rh_new = rw, rh + eyebrow_offset_right

            cv2.rectangle(frame, (lx_new, ly_new), (lx_new + lw_new, ly_new + lh_new), (0, 255, 255), FONT_THICKNESS)
            cv2.rectangle(frame, (rx_new, ry_new), (rx_new + rw_new, ry_new + rh_new), (0, 255, 255), FONT_THICKNESS)

            # -----------------------------
            # 5. Blink detection using the cropped eye images (grayscale with eyebrows)
            # -----------------------------
            left_eye_roi = gray[ly_new:ly_new + lh_new, lx_new:lx_new + lw_new]
            right_eye_roi = gray[ry_new:ry_new + rh_new, rx_new:rx_new + rw_new]

            eyes_open = False
            eye_state = "Unknown"
            if left_eye_roi.size != 0 and right_eye_roi.size != 0:
                try:
                    left_eye_resized = cv2.resize(left_eye_roi, EYE_INPUT_SIZE)
                    right_eye_resized = cv2.resize(right_eye_roi, EYE_INPUT_SIZE)
                except Exception as e:
                    left_eye_resized = None
                    right_eye_resized = None

                if left_eye_resized is not None and right_eye_resized is not None:
                    left_eye_input = left_eye_resized.flatten().astype("float32") / 255.0
                    right_eye_input = right_eye_resized.flatten().astype("float32") / 255.0

                    left_pred = blink_model.predict([left_eye_input])[0]
                    right_pred = blink_model.predict([right_eye_input])[0]

                    # Inverted logic: prediction of 1 indicates closed; 0 indicates open.
                    if left_pred == 1 and right_pred == 1:
                        eye_state = "Closed"
                        eyes_open = False
                    else:
                        eye_state = "Open"
                        eyes_open = True

                    cv2.putText(frame, f"Eye: {eye_state}", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
            else:
                eyes_open = False

         
            if looking:
                if eyes_open:
                    closed_eye_consecutive = 0
                    if good_start_time is None:
                        good_start_time = current_time
                else:
                    closed_eye_consecutive += 1
                    if closed_eye_consecutive &gt; MAX_CLOSED_FRAMES:
                        good_start_time = None
            else:
                good_start_time = None
                closed_eye_consecutive = 0

            # If criteria are met for 10 seconds and attendance not yet marked.
            if good_start_time is not None:
                elapsed = current_time - good_start_time
                cv2.putText(frame, f"Good for {elapsed:.1f}s", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
                if elapsed &gt;= ATTENDANCE_DURATION and not attendance_marked:
                    attendance_marked = True
                    attendance_marked_time = current_time
            else:
                cv2.putText(frame, "Reset Timer", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
        else:
            good_start_time = None
            closed_eye_consecutive = 0

        # -----------------------------
        # 7. Display the Attendance Marked message at the top-right (if within display duration)
        # -----------------------------
        if attendance_marked_time is not None:
            if current_time - attendance_marked_time &lt;= ATTENDANCE_MESSAGE_DURATION:
                # Calculate position at top right
                text = "Attendance Marked"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ATTENDANCE_FONT_SCALE, FONT_THICKNESS)
                # Place text with some margin from top-right corner
                pos = (frame.shape[1] - text_width - 20, text_height + 20)
                cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, ATTENDANCE_FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
            else:
                # After 2 seconds, clear the attendance display time.
                attendance_marked_time = None

        cv2.imshow("Face &amp; Eye Detection with Attendance", frame)
        if cv2.waitKey(1) &amp; 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Pass a video file path or 0 for webcam.
    detect_face_eyes(0)</textarea><button hidden="" data-testid="" data-hotkey="Alt+F1,Control+Alt+˙,Control+Alt+h" data-hotkey-scope="read-only-cursor-text-area"></button><div class="Box-sc-g0xbh4-0 kHHiZS"><div class="Box-sc-g0xbh4-0 jqUoVd react-code-line-container" tabindex="0"><div class="Box-sc-g0xbh4-0 cnUKpU react-code-file-contents" role="presentation" aria-hidden="true" data-tab-size="8" data-testid="code-lines-container" data-paste-markdown-skip="true" data-hpc="true" style="height: 4780px;"><div class="react-line-numbers" style="pointer-events: auto; height: 4780px; position: relative; z-index: 2;"><div data-line-number="1" class="react-line-number react-code-text" style="padding-right: 16px;">1</div><div data-line-number="2" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(20px);">2</div><div data-line-number="3" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(40px);">3</div><div data-line-number="4" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(60px);">4</div><div data-line-number="5" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(80px);">5</div><div data-line-number="6" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(100px);">6</div><div data-line-number="7" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(120px);">7</div><div data-line-number="8" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(140px);">8</div><div data-line-number="9" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(160px);">9</div><div data-line-number="10" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(180px);">10</div><div data-line-number="11" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(200px);">11</div><div data-line-number="12" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(220px);">12</div><div data-line-number="13" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(240px);">13</div><div data-line-number="14" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(260px);">14</div><div data-line-number="15" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(280px);">15</div><div data-line-number="16" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(300px);">16</div><div data-line-number="17" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(320px);">17</div><div data-line-number="18" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(340px);">18</div><div data-line-number="19" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(360px);">19</div><div data-line-number="20" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(380px);">20</div><div data-line-number="21" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(400px);">21</div><div data-line-number="22" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(420px);">22</div><div data-line-number="23" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(440px);">23</div><div data-line-number="24" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(460px);">24</div><div data-line-number="25" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(480px);">25</div><div data-line-number="26" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(500px);">26</div><div data-line-number="27" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(520px);">27</div><div data-line-number="28" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(540px);">28</div><div data-line-number="29" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(560px);">29</div><div data-line-number="30" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(580px);">30</div><div data-line-number="31" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(600px);">31</div><div data-line-number="32" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(620px);">32</div><div data-line-number="33" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(640px);">33</div><div data-line-number="34" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(660px);">34</div><div data-line-number="35" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(680px);">35</div><div data-line-number="36" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(700px);">36</div><div data-line-number="37" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(720px);">37</div><div data-line-number="38" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(740px);">38</div><div data-line-number="39" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(760px);">39</div><div data-line-number="40" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(780px);">40</div><div data-line-number="41" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(800px);">41</div><div data-line-number="42" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(820px);">42</div><div data-line-number="43" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(840px);">43<span class="Box-sc-g0xbh4-0 cJGaMs"><div aria-label="Collapse code section" role="button" class="Box-sc-g0xbh4-0 iGLarr"><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-down Octicon-sc-9kayk9-0" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path></svg></div></span></div><div data-line-number="44" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(860px);">44</div><div data-line-number="45" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(880px);">45</div><div data-line-number="46" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(900px);">46</div><div data-line-number="47" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(920px);">47</div><div data-line-number="48" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(940px);">48</div><div data-line-number="49" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(960px);">49</div><div data-line-number="50" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(980px);">50</div><div data-line-number="51" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1000px);">51</div><div data-line-number="52" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1020px);">52</div><div data-line-number="53" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1040px);">53</div><div data-line-number="54" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1060px);">54</div><div data-line-number="55" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1080px);">55</div><div data-line-number="56" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1100px);">56</div><div data-line-number="57" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1120px);">57</div><div data-line-number="58" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1140px);">58</div><div data-line-number="59" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1160px);">59</div><div data-line-number="60" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1180px);">60</div><div data-line-number="61" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1200px);">61</div><div data-line-number="62" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1220px);">62</div><div data-line-number="63" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1240px);">63</div><div data-line-number="64" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1260px);">64</div><div data-line-number="65" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1280px);">65</div><div data-line-number="66" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1300px);">66</div><div data-line-number="67" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1320px);">67</div><div data-line-number="68" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1340px);">68</div><div data-line-number="69" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1360px);">69</div><div data-line-number="70" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1380px);">70</div><div data-line-number="71" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1400px);">71</div><div data-line-number="72" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1420px);">72</div><div data-line-number="73" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1440px);">73</div><div data-line-number="74" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1460px);">74</div><div data-line-number="75" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1480px);">75</div><div data-line-number="76" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1500px);">76</div><div data-line-number="77" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1520px);">77</div><div data-line-number="78" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1540px);">78</div><div data-line-number="79" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1560px);">79</div><div data-line-number="80" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1580px);">80</div><div data-line-number="81" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1600px);">81</div><div data-line-number="82" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1620px);">82</div><div data-line-number="83" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1640px);">83</div><div data-line-number="84" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1660px);">84</div><div data-line-number="85" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1680px);">85</div><div data-line-number="86" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1700px);">86</div><div data-line-number="87" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1720px);">87</div><div data-line-number="88" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1740px);">88</div><div data-line-number="89" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1760px);">89</div><div data-line-number="90" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1780px);">90</div><div data-line-number="91" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1800px);">91</div><div data-line-number="92" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1820px);">92</div><div data-line-number="93" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1840px);">93</div><div data-line-number="94" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1860px);">94</div><div data-line-number="95" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1880px);">95</div><div data-line-number="96" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1900px);">96</div><div data-line-number="97" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1920px);">97</div><div data-line-number="98" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1940px);">98</div><div data-line-number="99" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1960px);">99</div><div data-line-number="100" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1980px);">100</div><div data-line-number="101" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2000px);">101</div><div data-line-number="102" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2020px);">102</div><div data-line-number="103" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2040px);">103</div><div data-line-number="104" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2060px);">104</div><div data-line-number="105" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2080px);">105</div><div data-line-number="106" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2100px);">106</div><div data-line-number="107" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2120px);">107</div><div data-line-number="108" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2140px);">108</div><div data-line-number="109" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2160px);">109</div><div data-line-number="110" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2180px);">110</div><div data-line-number="111" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2200px);">111</div><div data-line-number="112" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2220px);">112</div><div data-line-number="113" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2240px);">113</div><div data-line-number="114" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2260px);">114</div><div data-line-number="115" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2280px);">115</div><div data-line-number="116" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2300px);">116</div><div data-line-number="117" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2320px);">117</div><div data-line-number="118" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2340px);">118</div><div data-line-number="119" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2360px);">119</div><div data-line-number="166" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3300px);">166</div><div data-line-number="167" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3320px);">167</div><div data-line-number="168" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3340px);">168</div><div data-line-number="169" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3360px);">169</div><div data-line-number="170" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3380px);">170</div><div data-line-number="171" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3400px);">171</div><div data-line-number="172" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3420px);">172</div><div data-line-number="173" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3440px);">173</div><div data-line-number="174" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3460px);">174</div><div data-line-number="175" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3480px);">175</div><div data-line-number="176" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3500px);">176</div><div data-line-number="177" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3520px);">177</div><div data-line-number="178" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3540px);">178</div><div data-line-number="179" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3560px);">179</div><div data-line-number="180" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3580px);">180</div><div data-line-number="181" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3600px);">181</div><div data-line-number="182" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3620px);">182</div><div data-line-number="183" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3640px);">183</div><div data-line-number="184" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3660px);">184</div><div data-line-number="185" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3680px);">185</div><div data-line-number="186" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3700px);">186</div><div data-line-number="187" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3720px);">187</div><div data-line-number="188" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3740px);">188</div><div data-line-number="189" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3760px);">189</div><div data-line-number="190" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3780px);">190</div><div data-line-number="191" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3800px);">191</div><div data-line-number="192" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3820px);">192</div><div data-line-number="193" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3840px);">193</div><div data-line-number="194" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3860px);">194</div><div data-line-number="195" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3880px);">195</div><div data-line-number="196" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3900px);">196</div><div data-line-number="197" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3920px);">197</div><div data-line-number="198" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3940px);">198</div><div data-line-number="199" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3960px);">199</div><div data-line-number="200" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(3980px);">200</div><div data-line-number="201" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4000px);">201</div><div data-line-number="202" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4020px);">202</div><div data-line-number="203" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4040px);">203</div><div data-line-number="204" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4060px);">204</div><div data-line-number="205" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4080px);">205</div><div data-line-number="206" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4100px);">206</div><div data-line-number="207" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4120px);">207</div><div data-line-number="208" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4140px);">208</div><div data-line-number="209" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4160px);">209</div><div data-line-number="210" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4180px);">210</div><div data-line-number="211" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4200px);">211</div><div data-line-number="212" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4220px);">212</div><div data-line-number="213" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4240px);">213</div><div data-line-number="214" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4260px);">214</div><div data-line-number="215" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4280px);">215</div><div data-line-number="216" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4300px);">216</div><div data-line-number="217" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4320px);">217</div><div data-line-number="218" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4340px);">218</div><div data-line-number="219" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4360px);">219</div><div data-line-number="220" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4380px);">220</div><div data-line-number="221" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4400px);">221</div><div data-line-number="222" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4420px);">222</div><div data-line-number="223" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4440px);">223</div><div data-line-number="224" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4460px);">224</div><div data-line-number="225" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4480px);">225</div><div data-line-number="226" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4500px);">226</div><div data-line-number="227" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4520px);">227</div><div data-line-number="228" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4540px);">228</div><div data-line-number="229" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4560px);">229</div><div data-line-number="230" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4580px);">230</div><div data-line-number="231" class="child-of-line-42  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4600px);">231</div><div data-line-number="232" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4620px);">232</div><div data-line-number="233" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4640px);">233</div><div data-line-number="234" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4660px);">234</div><div data-line-number="235" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4680px);">235</div><div data-line-number="236" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4700px);">236</div><div data-line-number="237" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4720px);">237</div><div data-line-number="238" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4740px);">238</div><div data-line-number="239" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(4760px);">239</div></div><div class="react-code-lines" style="height: 4780px;"><div data-key="0" class="react-code-text react-code-line-contents" style="min-height: auto;"><div><div id="LC1" class="react-file-line html-div" data-testid="code-cell" data-line-number="1" inert="inert" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">cv2</span></div></div></div><div data-key="1" class="react-code-text react-code-line-contents virtual" style="transform: translateY(20px); min-height: auto;"><div><div id="LC2" class="react-file-line html-div" data-testid="code-cell" data-line-number="2" inert="inert" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">dlib</span></div></div></div><div data-key="2" class="react-code-text react-code-line-contents virtual" style="transform: translateY(40px); min-height: auto;"><div><div id="LC3" class="react-file-line html-div" data-testid="code-cell" data-line-number="3" inert="inert" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">numpy</span> <span class="pl-k">as</span> <span class="pl-s1">np</span></div></div></div><div data-key="3" class="react-code-text react-code-line-contents virtual" style="transform: translateY(60px); min-height: auto;"><div><div id="LC4" class="react-file-line html-div" data-testid="code-cell" data-line-number="4" inert="inert" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">time</span></div></div></div><div data-key="4" class="react-code-text react-code-line-contents virtual" style="transform: translateY(80px); min-height: auto;"><div><div id="LC5" class="react-file-line html-div" data-testid="code-cell" data-line-number="5" inert="inert" style="position: relative;"><span class="pl-k">from</span> <span class="pl-s1">tensorflow</span>.<span class="pl-s1">keras</span>.<span class="pl-s1">models</span> <span class="pl-k">import</span> <span class="pl-s1">load_model</span></div></div></div><div data-key="5" class="react-code-text react-code-line-contents virtual" style="transform: translateY(100px); min-height: auto;"><div><div id="LC6" class="react-file-line html-div" data-testid="code-cell" data-line-number="6" inert="inert" style="position: relative;"><span class="pl-k">from</span> <span class="pl-s1">joblib</span> <span class="pl-k">import</span> <span class="pl-s1">load</span> <span class="pl-k">as</span> <span class="pl-s1">joblib_load</span></div></div></div><div data-key="6" class="react-code-text react-code-line-contents virtual" style="transform: translateY(120px); min-height: auto;"><div><div id="LC7" class="react-file-line html-div" data-testid="code-cell" data-line-number="7" inert="inert" style="position: relative;">
</div></div></div><div data-key="7" class="react-code-text react-code-line-contents virtual" style="transform: translateY(140px); min-height: auto;"><div><div id="LC8" class="react-file-line html-div" data-testid="code-cell" data-line-number="8" inert="inert" style="position: relative;"><span class="pl-c"># -----------------------------</span></div></div></div><div data-key="8" class="react-code-text react-code-line-contents virtual" style="transform: translateY(160px); min-height: auto;"><div><div id="LC9" class="react-file-line html-div" data-testid="code-cell" data-line-number="9" inert="inert" style="position: relative;"><span class="pl-c"># Load Haar cascades and Dlib model</span></div></div></div><div data-key="9" class="react-code-text react-code-line-contents virtual" style="transform: translateY(180px); min-height: auto;"><div><div id="LC10" class="react-file-line html-div" data-testid="code-cell" data-line-number="10" inert="inert" style="position: relative;"><span class="pl-c"># -----------------------------</span></div></div></div><div data-key="10" class="react-code-text react-code-line-contents virtual" style="transform: translateY(200px); min-height: auto;"><div><div id="LC11" class="react-file-line html-div" data-testid="code-cell" data-line-number="11" inert="inert" style="position: relative;"><span class="pl-s1">face_cascade</span> <span class="pl-c1">=</span> <span class="pl-s1">cv2</span>.<span class="pl-c1">CascadeClassifier</span>(<span class="pl-s">"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/haarcascade_frontalface_default.xml"</span>)</div></div></div><div data-key="11" class="react-code-text react-code-line-contents virtual" style="transform: translateY(220px); min-height: auto;"><div><div id="LC12" class="react-file-line html-div" data-testid="code-cell" data-line-number="12" inert="inert" style="position: relative;"><span class="pl-s1">eye_cascade</span> <span class="pl-c1">=</span> <span class="pl-s1">cv2</span>.<span class="pl-c1">CascadeClassifier</span>(<span class="pl-s">"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/haarcascade_eye.xml"</span>)</div></div></div><div data-key="12" class="react-code-text react-code-line-contents virtual" style="transform: translateY(240px); min-height: auto;"><div><div id="LC13" class="react-file-line html-div" data-testid="code-cell" data-line-number="13" inert="inert" style="position: relative;"><span class="pl-s1">predictor_path</span> <span class="pl-c1">=</span> <span class="pl-s">"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/shape_predictor_68_face_landmarks.dat"</span></div></div></div><div data-key="13" class="react-code-text react-code-line-contents virtual" style="transform: translateY(260px); min-height: auto;"><div><div id="LC14" class="react-file-line html-div" data-testid="code-cell" data-line-number="14" inert="inert" style="position: relative;"><span class="pl-s1">detector</span> <span class="pl-c1">=</span> <span class="pl-s1">dlib</span>.<span class="pl-c1">get_frontal_face_detector</span>()</div></div></div><div data-key="14" class="react-code-text react-code-line-contents virtual" style="transform: translateY(280px); min-height: auto;"><div><div id="LC15" class="react-file-line html-div" data-testid="code-cell" data-line-number="15" inert="inert" style="position: relative;"><span class="pl-s1">predictor</span> <span class="pl-c1">=</span> <span class="pl-s1">dlib</span>.<span class="pl-c1">shape_predictor</span>(<span class="pl-s1">predictor_path</span>)</div></div></div><div data-key="15" class="react-code-text react-code-line-contents virtual" style="transform: translateY(300px); min-height: auto;"><div><div id="LC16" class="react-file-line html-div" data-testid="code-cell" data-line-number="16" inert="inert" style="position: relative;">
</div></div></div><div data-key="16" class="react-code-text react-code-line-contents virtual" style="transform: translateY(320px); min-height: auto;"><div><div id="LC17" class="react-file-line html-div" data-testid="code-cell" data-line-number="17" inert="inert" style="position: relative;"><span class="pl-c"># -----------------------------</span></div></div></div><div data-key="17" class="react-code-text react-code-line-contents virtual" style="transform: translateY(340px); min-height: auto;"><div><div id="LC18" class="react-file-line html-div" data-testid="code-cell" data-line-number="18" inert="inert" style="position: relative;"><span class="pl-c"># Load your two trained ML models</span></div></div></div><div data-key="18" class="react-code-text react-code-line-contents virtual" style="transform: translateY(360px); min-height: auto;"><div><div id="LC19" class="react-file-line html-div" data-testid="code-cell" data-line-number="19" inert="inert" style="position: relative;"><span class="pl-c"># -----------------------------</span></div></div></div><div data-key="19" class="react-code-text react-code-line-contents virtual" style="transform: translateY(380px); min-height: auto;"><div><div id="LC20" class="react-file-line html-div" data-testid="code-cell" data-line-number="20" inert="inert" style="position: relative;"><span class="pl-c"># 1. Fine tuned MobileNetV2 for looking / not looking</span></div></div></div><div data-key="20" class="react-code-text react-code-line-contents virtual" style="transform: translateY(400px); min-height: auto;"><div><div id="LC21" class="react-file-line html-div" data-testid="code-cell" data-line-number="21" inert="inert" style="position: relative;"><span class="pl-c">#    (expects full face color image resized to 224x224)</span></div></div></div><div data-key="21" class="react-code-text react-code-line-contents virtual" style="transform: translateY(420px); min-height: auto;"><div><div id="LC22" class="react-file-line html-div" data-testid="code-cell" data-line-number="22" inert="inert" style="position: relative;"><span class="pl-s1">looking_model_path</span> <span class="pl-c1">=</span> <span class="pl-s">"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/fine_tuned_mobilenetv2.h5"</span></div></div></div><div data-key="22" class="react-code-text react-code-line-contents virtual" style="transform: translateY(440px); min-height: auto;"><div><div id="LC23" class="react-file-line html-div" data-testid="code-cell" data-line-number="23" inert="inert" style="position: relative;"><span class="pl-s1">looking_model</span> <span class="pl-c1">=</span> <span class="pl-en">load_model</span>(<span class="pl-s1">looking_model_path</span>)</div></div></div><div data-key="23" class="react-code-text react-code-line-contents virtual" style="transform: translateY(460px); min-height: auto;"><div><div id="LC24" class="react-file-line html-div" data-testid="code-cell" data-line-number="24" inert="inert" style="position: relative;">
</div></div></div><div data-key="24" class="react-code-text react-code-line-contents virtual" style="transform: translateY(480px); min-height: auto;"><div><div id="LC25" class="react-file-line html-div" data-testid="code-cell" data-line-number="25" inert="inert" style="position: relative;"><span class="pl-c"># 2. Blink SVM for open/closed eye classification</span></div></div></div><div data-key="25" class="react-code-text react-code-line-contents virtual" style="transform: translateY(500px); min-height: auto;"><div><div id="LC26" class="react-file-line html-div" data-testid="code-cell" data-line-number="26" inert="inert" style="position: relative;"><span class="pl-c">#    (expects a flattened, normalized grayscale image of size 64x64, i.e. 4096 features)</span></div></div></div><div data-key="26" class="react-code-text react-code-line-contents virtual" style="transform: translateY(520px); min-height: auto;"><div><div id="LC27" class="react-file-line html-div" data-testid="code-cell" data-line-number="27" inert="inert" style="position: relative;"><span class="pl-s1">blink_model_path</span> <span class="pl-c1">=</span> <span class="pl-s">"/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Models/blink_svm.pkl"</span></div></div></div><div data-key="27" class="react-code-text react-code-line-contents virtual" style="transform: translateY(540px); min-height: auto;"><div><div id="LC28" class="react-file-line html-div" data-testid="code-cell" data-line-number="28" inert="inert" style="position: relative;"><span class="pl-s1">blink_model</span> <span class="pl-c1">=</span> <span class="pl-en">joblib_load</span>(<span class="pl-s1">blink_model_path</span>)</div></div></div><div data-key="28" class="react-code-text react-code-line-contents virtual" style="transform: translateY(560px); min-height: auto;"><div><div id="LC29" class="react-file-line html-div" data-testid="code-cell" data-line-number="29" inert="inert" style="position: relative;">
</div></div></div><div data-key="29" class="react-code-text react-code-line-contents virtual" style="transform: translateY(580px); min-height: auto;"><div><div id="LC30" class="react-file-line html-div" data-testid="code-cell" data-line-number="30" inert="inert" style="position: relative;"><span class="pl-c"># -----------------------------</span></div></div></div><div data-key="30" class="react-code-text react-code-line-contents virtual" style="transform: translateY(600px); min-height: auto;"><div><div id="LC31" class="react-file-line html-div" data-testid="code-cell" data-line-number="31" inert="inert" style="position: relative;"><span class="pl-c"># Global parameters for attendance logic and fonts</span></div></div></div><div data-key="31" class="react-code-text react-code-line-contents virtual" style="transform: translateY(620px); min-height: auto;"><div><div id="LC32" class="react-file-line html-div" data-testid="code-cell" data-line-number="32" inert="inert" style="position: relative;"><span class="pl-c"># -----------------------------</span></div></div></div><div data-key="32" class="react-code-text react-code-line-contents virtual" style="transform: translateY(640px); min-height: auto;"><div><div id="LC33" class="react-file-line html-div" data-testid="code-cell" data-line-number="33" inert="inert" style="position: relative;"><span class="pl-c1">ATTENDANCE_DURATION</span> <span class="pl-c1">=</span> <span class="pl-c1">10</span>       <span class="pl-c"># seconds required to mark attendance (must be "good" for 10 sec)</span></div></div></div><div data-key="33" class="react-code-text react-code-line-contents virtual" style="transform: translateY(660px); min-height: auto;"><div><div id="LC34" class="react-file-line html-div" data-testid="code-cell" data-line-number="34" inert="inert" style="position: relative;"><span class="pl-c1">ATTENDANCE_MESSAGE_DURATION</span> <span class="pl-c1">=</span> <span class="pl-c1">2</span>  <span class="pl-c"># seconds to display the attendance message</span></div></div></div><div data-key="34" class="react-code-text react-code-line-contents virtual" style="transform: translateY(680px); min-height: auto;"><div><div id="LC35" class="react-file-line html-div" data-testid="code-cell" data-line-number="35" inert="inert" style="position: relative;"><span class="pl-c1">MAX_CLOSED_FRAMES</span> <span class="pl-c1">=</span> <span class="pl-c1">10</span>         <span class="pl-c"># if eyes are closed for &gt; this many consecutive frames, reset the timer</span></div></div></div><div data-key="35" class="react-code-text react-code-line-contents virtual" style="transform: translateY(700px); min-height: auto;"><div><div id="LC36" class="react-file-line html-div" data-testid="code-cell" data-line-number="36" inert="inert" style="position: relative;"><span class="pl-c1">EYE_INPUT_SIZE</span> <span class="pl-c1">=</span> (<span class="pl-c1">64</span>, <span class="pl-c1">64</span>)      <span class="pl-c"># expected input size for the blink SVM (64x64 = 4096 features)</span></div></div></div><div data-key="36" class="react-code-text react-code-line-contents virtual" style="transform: translateY(720px); min-height: auto;"><div><div id="LC37" class="react-file-line html-div" data-testid="code-cell" data-line-number="37" inert="inert" style="position: relative;">
</div></div></div><div data-key="37" class="react-code-text react-code-line-contents virtual" style="transform: translateY(740px); min-height: auto;"><div><div id="LC38" class="react-file-line html-div" data-testid="code-cell" data-line-number="38" inert="inert" style="position: relative;"><span class="pl-c"># Font scales for on-screen text</span></div></div></div><div data-key="38" class="react-code-text react-code-line-contents virtual" style="transform: translateY(760px); min-height: auto;"><div><div id="LC39" class="react-file-line html-div" data-testid="code-cell" data-line-number="39" inert="inert" style="position: relative;"><span class="pl-c1">FONT_SCALE</span> <span class="pl-c1">=</span> <span class="pl-c1">1.2</span></div></div></div><div data-key="39" class="react-code-text react-code-line-contents virtual" style="transform: translateY(780px); min-height: auto;"><div><div id="LC40" class="react-file-line html-div" data-testid="code-cell" data-line-number="40" inert="inert" style="position: relative;"><span class="pl-c1">ATTENDANCE_FONT_SCALE</span> <span class="pl-c1">=</span> <span class="pl-c1">2.0</span></div></div></div><div data-key="40" class="react-code-text react-code-line-contents virtual" style="transform: translateY(800px); min-height: auto;"><div><div id="LC41" class="react-file-line html-div" data-testid="code-cell" data-line-number="41" inert="inert" style="position: relative;"><span class="pl-c1">FONT_THICKNESS</span> <span class="pl-c1">=</span> <span class="pl-c1">2</span></div></div></div><div data-key="41" class="react-code-text react-code-line-contents virtual" style="transform: translateY(820px); min-height: auto;"><div><div id="LC42" class="react-file-line html-div" data-testid="code-cell" data-line-number="42" inert="inert" style="position: relative;">
</div></div></div><div data-key="42" class="react-code-text react-code-line-contents virtual" style="transform: translateY(840px); min-height: auto;"><div><div id="LC43" class="react-file-line html-div" data-testid="code-cell" data-line-number="43" inert="inert" style="position: relative;"><span class="pl-k">def</span> <span class="pl-en">detect_face_eyes</span>(<span class="pl-s1">video_source</span><span class="pl-c1">=</span><span class="pl-c1">0</span>):</div></div></div><div data-key="43" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(860px); min-height: auto;"><div><div id="LC44" class="react-file-line html-div" data-testid="code-cell" data-line-number="44" inert="inert" style="position: relative;">    <span class="pl-s">"""</span></div></div></div><div data-key="44" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(880px); min-height: auto;"><div><div id="LC45" class="react-file-line html-div" data-testid="code-cell" data-line-number="45" inert="inert" style="position: relative;"><span class="pl-s">    Detects face, eyes, and landmarks; then runs the looking model and blink model.</span></div></div></div><div data-key="45" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(900px); min-height: auto;"><div><div id="LC46" class="react-file-line html-div" data-testid="code-cell" data-line-number="46" inert="inert" style="position: relative;"><span class="pl-s">    - The entire face region (in color) is passed to the looking model.</span></div></div></div><div data-key="46" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(920px); min-height: auto;"><div><div id="LC47" class="react-file-line html-div" data-testid="code-cell" data-line-number="47" inert="inert" style="position: relative;"><span class="pl-s">    - The eye regions (cropped from the grayscale image, expanded upward to include eyebrows)</span></div></div></div><div data-key="47" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(940px); min-height: auto;"><div><div id="LC48" class="react-file-line html-div" data-testid="code-cell" data-line-number="48" inert="inert" style="position: relative;"><span class="pl-s">      are resized to 64x64 and passed to the blink SVM.</span></div></div></div><div data-key="48" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(960px); min-height: auto;"><div><div id="LC49" class="react-file-line html-div" data-testid="code-cell" data-line-number="49" inert="inert" style="position: relative;"><span class="pl-s">      </span></div></div></div><div data-key="49" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(980px); min-height: auto;"><div><div id="LC50" class="react-file-line html-div" data-testid="code-cell" data-line-number="50" inert="inert" style="position: relative;"><span class="pl-s">    Prediction logic (inverted if necessary):</span></div></div></div><div data-key="50" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1000px); min-height: auto;"><div><div id="LC51" class="react-file-line html-div" data-testid="code-cell" data-line-number="51" inert="inert" style="position: relative;"><span class="pl-s">      * For the looking model, a probability &lt;= 0.5 indicates "looking."</span></div></div></div><div data-key="51" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1020px); min-height: auto;"><div><div id="LC52" class="react-file-line html-div" data-testid="code-cell" data-line-number="52" inert="inert" style="position: relative;"><span class="pl-s">      * For the blink model, a prediction of 1 indicates "closed" and 0 indicates "open."</span></div></div></div><div data-key="52" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1040px); min-height: auto;"><div><div id="LC53" class="react-file-line html-div" data-testid="code-cell" data-line-number="53" inert="inert" style="position: relative;"><span class="pl-s">      </span></div></div></div><div data-key="53" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1060px); min-height: auto;"><div><div id="LC54" class="react-file-line html-div" data-testid="code-cell" data-line-number="54" inert="inert" style="position: relative;"><span class="pl-s">    If a face is detected, the system checks if the subject is looking. If yes,</span></div></div></div><div data-key="54" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1080px); min-height: auto;"><div><div id="LC55" class="react-file-line html-div" data-testid="code-cell" data-line-number="55" inert="inert" style="position: relative;"><span class="pl-s">    it then checks the eyes. When both eyes are predicted as open (blink SVM returns 0)</span></div></div></div><div data-key="55" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1100px); min-height: auto;"><div><div id="LC56" class="react-file-line html-div" data-testid="code-cell" data-line-number="56" inert="inert" style="position: relative;"><span class="pl-s">    for a continuous period (ATTENDANCE_DURATION seconds) with only brief closures allowed,</span></div></div></div><div data-key="56" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1120px); min-height: auto;"><div><div id="LC57" class="react-file-line html-div" data-testid="code-cell" data-line-number="57" inert="inert" style="position: relative;"><span class="pl-s">    attendance is marked.</span></div></div></div><div data-key="57" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1140px); min-height: auto;"><div><div id="LC58" class="react-file-line html-div" data-testid="code-cell" data-line-number="58" inert="inert" style="position: relative;"><span class="pl-s">    """</span></div></div></div><div data-key="58" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1160px); min-height: auto;"><div><div id="LC59" class="react-file-line html-div" data-testid="code-cell" data-line-number="59" inert="inert" style="position: relative;">    <span class="pl-s1">cap</span> <span class="pl-c1">=</span> <span class="pl-s1">cv2</span>.<span class="pl-c1">VideoCapture</span>(<span class="pl-s1">video_source</span>)</div></div></div><div data-key="59" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1180px); min-height: auto;"><div><div id="LC60" class="react-file-line html-div" data-testid="code-cell" data-line-number="60" inert="inert" style="position: relative;">    <span class="pl-k">if</span> <span class="pl-c1">not</span> <span class="pl-s1">cap</span>.<span class="pl-c1">isOpened</span>():</div></div></div><div data-key="60" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1200px); min-height: auto;"><div><div id="LC61" class="react-file-line html-div" data-testid="code-cell" data-line-number="61" inert="inert" style="position: relative;">        <span class="pl-en">print</span>(<span class="pl-s">"Error: Unable to open video source."</span>)</div></div></div><div data-key="61" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1220px); min-height: auto;"><div><div id="LC62" class="react-file-line html-div" data-testid="code-cell" data-line-number="62" inert="inert" style="position: relative;">        <span class="pl-k">return</span></div></div></div><div data-key="62" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1240px); min-height: auto;"><div><div id="LC63" class="react-file-line html-div" data-testid="code-cell" data-line-number="63" inert="inert" style="position: relative;">
</div></div></div><div data-key="63" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1260px); min-height: auto;"><div><div id="LC64" class="react-file-line html-div" data-testid="code-cell" data-line-number="64" inert="inert" style="position: relative;">    <span class="pl-c"># Variables for attendance logic</span></div></div></div><div data-key="64" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1280px); min-height: auto;"><div><div id="LC65" class="react-file-line html-div" data-testid="code-cell" data-line-number="65" inert="inert" style="position: relative;">    <span class="pl-s1">good_start_time</span> <span class="pl-c1">=</span> <span class="pl-c1">None</span>       <span class="pl-c"># Time when "good" criteria first started</span></div></div></div><div data-key="65" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1300px); min-height: auto;"><div><div id="LC66" class="react-file-line html-div" data-testid="code-cell" data-line-number="66" inert="inert" style="position: relative;">    <span class="pl-s1">attendance_marked</span> <span class="pl-c1">=</span> <span class="pl-c1">False</span>    <span class="pl-c"># Whether attendance has been marked already</span></div></div></div><div data-key="66" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1320px); min-height: auto;"><div><div id="LC67" class="react-file-line html-div" data-testid="code-cell" data-line-number="67" inert="inert" style="position: relative;">    <span class="pl-s1">attendance_marked_time</span> <span class="pl-c1">=</span> <span class="pl-c1">None</span>  <span class="pl-c"># Time at which attendance was marked (for display)</span></div></div></div><div data-key="67" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1340px); min-height: auto;"><div><div id="LC68" class="react-file-line html-div" data-testid="code-cell" data-line-number="68" inert="inert" style="position: relative;">    <span class="pl-s1">closed_eye_consecutive</span> <span class="pl-c1">=</span> <span class="pl-c1">0</span>   <span class="pl-c"># Count of consecutive frames with eyes closed</span></div></div></div><div data-key="68" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1360px); min-height: auto;"><div><div id="LC69" class="react-file-line html-div" data-testid="code-cell" data-line-number="69" inert="inert" style="position: relative;">
</div></div></div><div data-key="69" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1380px); min-height: auto;"><div><div id="LC70" class="react-file-line html-div" data-testid="code-cell" data-line-number="70" inert="inert" style="position: relative;">    <span class="pl-k">while</span> <span class="pl-s1">cap</span>.<span class="pl-c1">isOpened</span>():</div></div></div><div data-key="70" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1400px); min-height: auto;"><div><div id="LC71" class="react-file-line html-div" data-testid="code-cell" data-line-number="71" inert="inert" style="position: relative;">        <span class="pl-s1">ret</span>, <span class="pl-s1">frame</span> <span class="pl-c1">=</span> <span class="pl-s1">cap</span>.<span class="pl-c1">read</span>()</div></div></div><div data-key="71" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1420px); min-height: auto;"><div><div id="LC72" class="react-file-line html-div" data-testid="code-cell" data-line-number="72" inert="inert" style="position: relative;">        <span class="pl-k">if</span> <span class="pl-c1">not</span> <span class="pl-s1">ret</span>:</div></div></div><div data-key="72" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1440px); min-height: auto;"><div><div id="LC73" class="react-file-line html-div" data-testid="code-cell" data-line-number="73" inert="inert" style="position: relative;">            <span class="pl-k">break</span></div></div></div><div data-key="73" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1460px); min-height: auto;"><div><div id="LC74" class="react-file-line html-div" data-testid="code-cell" data-line-number="74" inert="inert" style="position: relative;">
</div></div></div><div data-key="74" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1480px); min-height: auto;"><div><div id="LC75" class="react-file-line html-div" data-testid="code-cell" data-line-number="75" inert="inert" style="position: relative;">        <span class="pl-s1">current_time</span> <span class="pl-c1">=</span> <span class="pl-s1">time</span>.<span class="pl-c1">time</span>()</div></div></div><div data-key="75" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1500px); min-height: auto;"><div><div id="LC76" class="react-file-line html-div" data-testid="code-cell" data-line-number="76" inert="inert" style="position: relative;">        <span class="pl-s1">gray</span> <span class="pl-c1">=</span> <span class="pl-s1">cv2</span>.<span class="pl-c1">cvtColor</span>(<span class="pl-s1">frame</span>, <span class="pl-s1">cv2</span>.<span class="pl-c1">COLOR_BGR2GRAY</span>)</div></div></div><div data-key="76" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1520px); min-height: auto;"><div><div id="LC77" class="react-file-line html-div" data-testid="code-cell" data-line-number="77" inert="inert" style="position: relative;">
</div></div></div><div data-key="77" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1540px); min-height: auto;"><div><div id="LC78" class="react-file-line html-div" data-testid="code-cell" data-line-number="78" inert="inert" style="position: relative;">        <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="78" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1560px); min-height: auto;"><div><div id="LC79" class="react-file-line html-div" data-testid="code-cell" data-line-number="79" inert="inert" style="position: relative;">        <span class="pl-c"># 1. Face detection using Haar cascade</span></div></div></div><div data-key="79" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1580px); min-height: auto;"><div><div id="LC80" class="react-file-line html-div" data-testid="code-cell" data-line-number="80" inert="inert" style="position: relative;">        <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="80" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1600px); min-height: auto;"><div><div id="LC81" class="react-file-line html-div" data-testid="code-cell" data-line-number="81" inert="inert" style="position: relative;">        <span class="pl-s1">faces</span> <span class="pl-c1">=</span> <span class="pl-s1">face_cascade</span>.<span class="pl-c1">detectMultiScale</span>(<span class="pl-s1">gray</span>, <span class="pl-s1">scaleFactor</span><span class="pl-c1">=</span><span class="pl-c1">1.3</span>, <span class="pl-s1">minNeighbors</span><span class="pl-c1">=</span><span class="pl-c1">5</span>)</div></div></div><div data-key="81" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1620px); min-height: auto;"><div><div id="LC82" class="react-file-line html-div" data-testid="code-cell" data-line-number="82" inert="inert" style="position: relative;">        <span class="pl-k">if</span> <span class="pl-en">len</span>(<span class="pl-s1">faces</span>) <span class="pl-c1">&gt;</span> <span class="pl-c1">0</span>:</div></div></div><div data-key="82" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1640px); min-height: auto;"><div><div id="LC83" class="react-file-line html-div" data-testid="code-cell" data-line-number="83" inert="inert" style="position: relative;">            <span class="pl-c"># For simplicity, select the largest detected face.</span></div></div></div><div data-key="83" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1660px); min-height: auto;"><div><div id="LC84" class="react-file-line html-div" data-testid="code-cell" data-line-number="84" inert="inert" style="position: relative;">            (<span class="pl-s1">x</span>, <span class="pl-s1">y</span>, <span class="pl-s1">w</span>, <span class="pl-s1">h</span>) <span class="pl-c1">=</span> <span class="pl-en">max</span>(<span class="pl-s1">faces</span>, <span class="pl-s1">key</span><span class="pl-c1">=</span><span class="pl-k">lambda</span> <span class="pl-s1">r</span>: <span class="pl-s1">r</span>[<span class="pl-c1">2</span>] <span class="pl-c1">*</span> <span class="pl-s1">r</span>[<span class="pl-c1">3</span>])</div></div></div><div data-key="84" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1680px); min-height: auto;"><div><div id="LC85" class="react-file-line html-div" data-testid="code-cell" data-line-number="85" inert="inert" style="position: relative;">            <span class="pl-s1">cv2</span>.<span class="pl-c1">rectangle</span>(<span class="pl-s1">frame</span>, (<span class="pl-s1">x</span>, <span class="pl-s1">y</span>), (<span class="pl-s1">x</span> <span class="pl-c1">+</span> <span class="pl-s1">w</span>, <span class="pl-s1">y</span> <span class="pl-c1">+</span> <span class="pl-s1">h</span>), (<span class="pl-c1">255</span>, <span class="pl-c1">0</span>, <span class="pl-c1">0</span>), <span class="pl-c1">FONT_THICKNESS</span>)</div></div></div><div data-key="85" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1700px); min-height: auto;"><div><div id="LC86" class="react-file-line html-div" data-testid="code-cell" data-line-number="86" inert="inert" style="position: relative;">
</div></div></div><div data-key="86" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1720px); min-height: auto;"><div><div id="LC87" class="react-file-line html-div" data-testid="code-cell" data-line-number="87" inert="inert" style="position: relative;">            <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="87" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1740px); min-height: auto;"><div><div id="LC88" class="react-file-line html-div" data-testid="code-cell" data-line-number="88" inert="inert" style="position: relative;">            <span class="pl-c"># 2. Looking detection using the entire face crop</span></div></div></div><div data-key="88" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1760px); min-height: auto;"><div><div id="LC89" class="react-file-line html-div" data-testid="code-cell" data-line-number="89" inert="inert" style="position: relative;">            <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="89" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1780px); min-height: auto;"><div><div id="LC90" class="react-file-line html-div" data-testid="code-cell" data-line-number="90" inert="inert" style="position: relative;">            <span class="pl-s1">face_roi</span> <span class="pl-c1">=</span> <span class="pl-s1">frame</span>[<span class="pl-s1">y</span>:<span class="pl-s1">y</span> <span class="pl-c1">+</span> <span class="pl-s1">h</span>, <span class="pl-s1">x</span>:<span class="pl-s1">x</span> <span class="pl-c1">+</span> <span class="pl-s1">w</span>]</div></div></div><div data-key="90" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1800px); min-height: auto;"><div><div id="LC91" class="react-file-line html-div" data-testid="code-cell" data-line-number="91" inert="inert" style="position: relative;">            <span class="pl-k">try</span>:</div></div></div><div data-key="91" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1820px); min-height: auto;"><div><div id="LC92" class="react-file-line html-div" data-testid="code-cell" data-line-number="92" inert="inert" style="position: relative;">                <span class="pl-s1">face_roi_resized</span> <span class="pl-c1">=</span> <span class="pl-s1">cv2</span>.<span class="pl-c1">resize</span>(<span class="pl-s1">face_roi</span>, (<span class="pl-c1">224</span>, <span class="pl-c1">224</span>))</div></div></div><div data-key="92" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1840px); min-height: auto;"><div><div id="LC93" class="react-file-line html-div" data-testid="code-cell" data-line-number="93" inert="inert" style="position: relative;">            <span class="pl-k">except</span> <span class="pl-v">Exception</span> <span class="pl-k">as</span> <span class="pl-s1">e</span>:</div></div></div><div data-key="93" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1860px); min-height: auto;"><div><div id="LC94" class="react-file-line html-div" data-testid="code-cell" data-line-number="94" inert="inert" style="position: relative;">                <span class="pl-s1">face_roi_resized</span> <span class="pl-c1">=</span> <span class="pl-s1">face_roi</span></div></div></div><div data-key="94" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1880px); min-height: auto;"><div><div id="LC95" class="react-file-line html-div" data-testid="code-cell" data-line-number="95" inert="inert" style="position: relative;">            <span class="pl-s1">face_roi_norm</span> <span class="pl-c1">=</span> <span class="pl-s1">face_roi_resized</span>.<span class="pl-c1">astype</span>(<span class="pl-s">"float32"</span>) <span class="pl-c1">/</span> <span class="pl-c1">255.0</span></div></div></div><div data-key="95" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1900px); min-height: auto;"><div><div id="LC96" class="react-file-line html-div" data-testid="code-cell" data-line-number="96" inert="inert" style="position: relative;">            <span class="pl-s1">face_input</span> <span class="pl-c1">=</span> <span class="pl-s1">np</span>.<span class="pl-c1">expand_dims</span>(<span class="pl-s1">face_roi_norm</span>, <span class="pl-s1">axis</span><span class="pl-c1">=</span><span class="pl-c1">0</span>)</div></div></div><div data-key="96" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1920px); min-height: auto;"><div><div id="LC97" class="react-file-line html-div" data-testid="code-cell" data-line-number="97" inert="inert" style="position: relative;">            <span class="pl-s1">looking_pred</span> <span class="pl-c1">=</span> <span class="pl-s1">looking_model</span>.<span class="pl-c1">predict</span>(<span class="pl-s1">face_input</span>)</div></div></div><div data-key="97" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1940px); min-height: auto;"><div><div id="LC98" class="react-file-line html-div" data-testid="code-cell" data-line-number="98" inert="inert" style="position: relative;">            <span class="pl-c"># Inverted logic: probability &lt;= 0.5 means "looking"</span></div></div></div><div data-key="98" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1960px); min-height: auto;"><div><div id="LC99" class="react-file-line html-div" data-testid="code-cell" data-line-number="99" inert="inert" style="position: relative;">            <span class="pl-k">if</span> <span class="pl-s1">looking_pred</span>[<span class="pl-c1">0</span>][<span class="pl-c1">0</span>] <span class="pl-c1">&lt;=</span> <span class="pl-c1">0.5</span>:</div></div></div><div data-key="99" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(1980px); min-height: auto;"><div><div id="LC100" class="react-file-line html-div" data-testid="code-cell" data-line-number="100" inert="inert" style="position: relative;">                <span class="pl-s1">looking</span> <span class="pl-c1">=</span> <span class="pl-c1">True</span></div></div></div><div data-key="100" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2000px); min-height: auto;"><div><div id="LC101" class="react-file-line html-div" data-testid="code-cell" data-line-number="101" inert="inert" style="position: relative;">                <span class="pl-s1">looking_text</span> <span class="pl-c1">=</span> <span class="pl-s">"Looking: Yes"</span></div></div></div><div data-key="101" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2020px); min-height: auto;"><div><div id="LC102" class="react-file-line html-div" data-testid="code-cell" data-line-number="102" inert="inert" style="position: relative;">            <span class="pl-k">else</span>:</div></div></div><div data-key="102" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2040px); min-height: auto;"><div><div id="LC103" class="react-file-line html-div" data-testid="code-cell" data-line-number="103" inert="inert" style="position: relative;">                <span class="pl-s1">looking</span> <span class="pl-c1">=</span> <span class="pl-c1">False</span></div></div></div><div data-key="103" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2060px); min-height: auto;"><div><div id="LC104" class="react-file-line html-div" data-testid="code-cell" data-line-number="104" inert="inert" style="position: relative;">                <span class="pl-s1">looking_text</span> <span class="pl-c1">=</span> <span class="pl-s">"Looking: No"</span></div></div></div><div data-key="104" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2080px); min-height: auto;"><div><div id="LC105" class="react-file-line html-div" data-testid="code-cell" data-line-number="105" inert="inert" style="position: relative;">            <span class="pl-s1">cv2</span>.<span class="pl-c1">putText</span>(<span class="pl-s1">frame</span>, <span class="pl-s1">looking_text</span>, (<span class="pl-s1">x</span>, <span class="pl-s1">y</span> <span class="pl-c1">-</span> <span class="pl-c1">10</span>),</div></div></div><div data-key="105" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2100px); min-height: auto;"><div><div id="LC106" class="react-file-line html-div" data-testid="code-cell" data-line-number="106" inert="inert" style="position: relative;">                        <span class="pl-s1">cv2</span>.<span class="pl-c1">FONT_HERSHEY_SIMPLEX</span>, <span class="pl-c1">FONT_SCALE</span>, (<span class="pl-c1">255</span>, <span class="pl-c1">0</span>, <span class="pl-c1">0</span>), <span class="pl-c1">FONT_THICKNESS</span>)</div></div></div><div data-key="106" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2120px); min-height: auto;"><div><div id="LC107" class="react-file-line html-div" data-testid="code-cell" data-line-number="107" inert="inert" style="position: relative;">
</div></div></div><div data-key="107" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2140px); min-height: auto;"><div><div id="LC108" class="react-file-line html-div" data-testid="code-cell" data-line-number="108" inert="inert" style="position: relative;">            <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="108" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2160px); min-height: auto;"><div><div id="LC109" class="react-file-line html-div" data-testid="code-cell" data-line-number="109" inert="inert" style="position: relative;">            <span class="pl-c"># 3. Eye detection using Haar cascade within the face ROI</span></div></div></div><div data-key="109" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2180px); min-height: auto;"><div><div id="LC110" class="react-file-line html-div" data-testid="code-cell" data-line-number="110" inert="inert" style="position: relative;">            <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="110" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2200px); min-height: auto;"><div><div id="LC111" class="react-file-line html-div" data-testid="code-cell" data-line-number="111" inert="inert" style="position: relative;">            <span class="pl-s1">roi_gray</span> <span class="pl-c1">=</span> <span class="pl-s1">gray</span>[<span class="pl-s1">y</span>:<span class="pl-s1">y</span> <span class="pl-c1">+</span> <span class="pl-s1">h</span>, <span class="pl-s1">x</span>:<span class="pl-s1">x</span> <span class="pl-c1">+</span> <span class="pl-s1">w</span>]</div></div></div><div data-key="111" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2220px); min-height: auto;"><div><div id="LC112" class="react-file-line html-div" data-testid="code-cell" data-line-number="112" inert="inert" style="position: relative;">            <span class="pl-s1">roi_color</span> <span class="pl-c1">=</span> <span class="pl-s1">frame</span>[<span class="pl-s1">y</span>:<span class="pl-s1">y</span> <span class="pl-c1">+</span> <span class="pl-s1">h</span>, <span class="pl-s1">x</span>:<span class="pl-s1">x</span> <span class="pl-c1">+</span> <span class="pl-s1">w</span>]</div></div></div><div data-key="112" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2240px); min-height: auto;"><div><div id="LC113" class="react-file-line html-div" data-testid="code-cell" data-line-number="113" inert="inert" style="position: relative;">            <span class="pl-s1">eyes</span> <span class="pl-c1">=</span> <span class="pl-s1">eye_cascade</span>.<span class="pl-c1">detectMultiScale</span>(<span class="pl-s1">roi_gray</span>)</div></div></div><div data-key="113" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2260px); min-height: auto;"><div><div id="LC114" class="react-file-line html-div" data-testid="code-cell" data-line-number="114" inert="inert" style="position: relative;">            <span class="pl-k">for</span> (<span class="pl-s1">ex</span>, <span class="pl-s1">ey</span>, <span class="pl-s1">ew</span>, <span class="pl-s1">eh</span>) <span class="pl-c1">in</span> <span class="pl-s1">eyes</span>:</div></div></div><div data-key="114" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2280px); min-height: auto;"><div><div id="LC115" class="react-file-line html-div" data-testid="code-cell" data-line-number="115" inert="inert" style="position: relative;">                <span class="pl-s1">cv2</span>.<span class="pl-c1">rectangle</span>(<span class="pl-s1">roi_color</span>, (<span class="pl-s1">ex</span>, <span class="pl-s1">ey</span>), (<span class="pl-s1">ex</span> <span class="pl-c1">+</span> <span class="pl-s1">ew</span>, <span class="pl-s1">ey</span> <span class="pl-c1">+</span> <span class="pl-s1">eh</span>), (<span class="pl-c1">0</span>, <span class="pl-c1">255</span>, <span class="pl-c1">0</span>), <span class="pl-c1">FONT_THICKNESS</span>)</div></div></div><div data-key="115" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2300px); min-height: auto;"><div><div id="LC116" class="react-file-line html-div" data-testid="code-cell" data-line-number="116" inert="inert" style="position: relative;">
</div></div></div><div data-key="116" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2320px); min-height: auto;"><div><div id="LC117" class="react-file-line html-div" data-testid="code-cell" data-line-number="117" inert="inert" style="position: relative;">            <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="117" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2340px); min-height: auto;"><div><div id="LC118" class="react-file-line html-div" data-testid="code-cell" data-line-number="118" inert="inert" style="position: relative;">            <span class="pl-c"># 4. Get accurate eye coordinates using dlib’s landmarks</span></div></div></div><div data-key="118" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(2360px); min-height: auto;"><div><div id="LC119" class="react-file-line html-div" data-testid="code-cell" data-line-number="119" inert="inert" style="position: relative;">            <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="165" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3300px); min-height: auto;"><div><div id="LC166" class="react-file-line html-div" data-testid="code-cell" data-line-number="166" inert="inert" style="position: relative;">
</div></div></div><div data-key="166" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3320px); min-height: auto;"><div><div id="LC167" class="react-file-line html-div" data-testid="code-cell" data-line-number="167" inert="inert" style="position: relative;">                    <span class="pl-s1">left_pred</span> <span class="pl-c1">=</span> <span class="pl-s1">blink_model</span>.<span class="pl-c1">predict</span>([<span class="pl-s1">left_eye_input</span>])[<span class="pl-c1">0</span>]</div></div></div><div data-key="167" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3340px); min-height: auto;"><div><div id="LC168" class="react-file-line html-div" data-testid="code-cell" data-line-number="168" inert="inert" style="position: relative;">                    <span class="pl-s1">right_pred</span> <span class="pl-c1">=</span> <span class="pl-s1">blink_model</span>.<span class="pl-c1">predict</span>([<span class="pl-s1">right_eye_input</span>])[<span class="pl-c1">0</span>]</div></div></div><div data-key="168" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3360px); min-height: auto;"><div><div id="LC169" class="react-file-line html-div" data-testid="code-cell" data-line-number="169" inert="inert" style="position: relative;">
</div></div></div><div data-key="169" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3380px); min-height: auto;"><div><div id="LC170" class="react-file-line html-div" data-testid="code-cell" data-line-number="170" inert="inert" style="position: relative;">                    <span class="pl-c"># Inverted logic: prediction of 1 indicates closed; 0 indicates open.</span></div></div></div><div data-key="170" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3400px); min-height: auto;"><div><div id="LC171" class="react-file-line html-div" data-testid="code-cell" data-line-number="171" inert="inert" style="position: relative;">                    <span class="pl-k">if</span> <span class="pl-s1">left_pred</span> <span class="pl-c1">==</span> <span class="pl-c1">1</span> <span class="pl-c1">and</span> <span class="pl-s1">right_pred</span> <span class="pl-c1">==</span> <span class="pl-c1">1</span>:</div></div></div><div data-key="171" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3420px); min-height: auto;"><div><div id="LC172" class="react-file-line html-div" data-testid="code-cell" data-line-number="172" inert="inert" style="position: relative;">                        <span class="pl-s1">eye_state</span> <span class="pl-c1">=</span> <span class="pl-s">"Closed"</span></div></div></div><div data-key="172" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3440px); min-height: auto;"><div><div id="LC173" class="react-file-line html-div" data-testid="code-cell" data-line-number="173" inert="inert" style="position: relative;">                        <span class="pl-s1">eyes_open</span> <span class="pl-c1">=</span> <span class="pl-c1">False</span></div></div></div><div data-key="173" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3460px); min-height: auto;"><div><div id="LC174" class="react-file-line html-div" data-testid="code-cell" data-line-number="174" inert="inert" style="position: relative;">                    <span class="pl-k">else</span>:</div></div></div><div data-key="174" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3480px); min-height: auto;"><div><div id="LC175" class="react-file-line html-div" data-testid="code-cell" data-line-number="175" inert="inert" style="position: relative;">                        <span class="pl-s1">eye_state</span> <span class="pl-c1">=</span> <span class="pl-s">"Open"</span></div></div></div><div data-key="175" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3500px); min-height: auto;"><div><div id="LC176" class="react-file-line html-div" data-testid="code-cell" data-line-number="176" inert="inert" style="position: relative;">                        <span class="pl-s1">eyes_open</span> <span class="pl-c1">=</span> <span class="pl-c1">True</span></div></div></div><div data-key="176" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3520px); min-height: auto;"><div><div id="LC177" class="react-file-line html-div" data-testid="code-cell" data-line-number="177" inert="inert" style="position: relative;">
</div></div></div><div data-key="177" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3540px); min-height: auto;"><div><div id="LC178" class="react-file-line html-div" data-testid="code-cell" data-line-number="178" inert="inert" style="position: relative;">                    <span class="pl-s1">cv2</span>.<span class="pl-c1">putText</span>(<span class="pl-s1">frame</span>, <span class="pl-s">f"Eye: <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">eye_state</span><span class="pl-kos">}</span></span>"</span>, (<span class="pl-s1">x</span>, <span class="pl-s1">y</span> <span class="pl-c1">+</span> <span class="pl-s1">h</span> <span class="pl-c1">+</span> <span class="pl-c1">20</span>),</div></div></div><div data-key="178" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3560px); min-height: auto;"><div><div id="LC179" class="react-file-line html-div" data-testid="code-cell" data-line-number="179" inert="inert" style="position: relative;">                                <span class="pl-s1">cv2</span>.<span class="pl-c1">FONT_HERSHEY_SIMPLEX</span>, <span class="pl-c1">FONT_SCALE</span>, (<span class="pl-c1">0</span>, <span class="pl-c1">255</span>, <span class="pl-c1">255</span>), <span class="pl-c1">FONT_THICKNESS</span>)</div></div></div><div data-key="179" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3580px); min-height: auto;"><div><div id="LC180" class="react-file-line html-div" data-testid="code-cell" data-line-number="180" inert="inert" style="position: relative;">            <span class="pl-k">else</span>:</div></div></div><div data-key="180" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3600px); min-height: auto;"><div><div id="LC181" class="react-file-line html-div" data-testid="code-cell" data-line-number="181" inert="inert" style="position: relative;">                <span class="pl-s1">eyes_open</span> <span class="pl-c1">=</span> <span class="pl-c1">False</span></div></div></div><div data-key="181" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3620px); min-height: auto;"><div><div id="LC182" class="react-file-line html-div" data-testid="code-cell" data-line-number="182" inert="inert" style="position: relative;">
</div></div></div><div data-key="182" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3640px); min-height: auto;"><div><div id="LC183" class="react-file-line html-div" data-testid="code-cell" data-line-number="183" inert="inert" style="position: relative;">         </div></div></div><div data-key="183" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3660px); min-height: auto;"><div><div id="LC184" class="react-file-line html-div" data-testid="code-cell" data-line-number="184" inert="inert" style="position: relative;">            <span class="pl-k">if</span> <span class="pl-s1">looking</span>:</div></div></div><div data-key="184" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3680px); min-height: auto;"><div><div id="LC185" class="react-file-line html-div" data-testid="code-cell" data-line-number="185" inert="inert" style="position: relative;">                <span class="pl-k">if</span> <span class="pl-s1">eyes_open</span>:</div></div></div><div data-key="185" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3700px); min-height: auto;"><div><div id="LC186" class="react-file-line html-div" data-testid="code-cell" data-line-number="186" inert="inert" style="position: relative;">                    <span class="pl-s1">closed_eye_consecutive</span> <span class="pl-c1">=</span> <span class="pl-c1">0</span></div></div></div><div data-key="186" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3720px); min-height: auto;"><div><div id="LC187" class="react-file-line html-div" data-testid="code-cell" data-line-number="187" inert="inert" style="position: relative;">                    <span class="pl-k">if</span> <span class="pl-s1">good_start_time</span> <span class="pl-c1">is</span> <span class="pl-c1">None</span>:</div></div></div><div data-key="187" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3740px); min-height: auto;"><div><div id="LC188" class="react-file-line html-div" data-testid="code-cell" data-line-number="188" inert="inert" style="position: relative;">                        <span class="pl-s1">good_start_time</span> <span class="pl-c1">=</span> <span class="pl-s1">current_time</span></div></div></div><div data-key="188" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3760px); min-height: auto;"><div><div id="LC189" class="react-file-line html-div" data-testid="code-cell" data-line-number="189" inert="inert" style="position: relative;">                <span class="pl-k">else</span>:</div></div></div><div data-key="189" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3780px); min-height: auto;"><div><div id="LC190" class="react-file-line html-div" data-testid="code-cell" data-line-number="190" inert="inert" style="position: relative;">                    <span class="pl-s1">closed_eye_consecutive</span> <span class="pl-c1">+=</span> <span class="pl-c1">1</span></div></div></div><div data-key="190" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3800px); min-height: auto;"><div><div id="LC191" class="react-file-line html-div" data-testid="code-cell" data-line-number="191" inert="inert" style="position: relative;">                    <span class="pl-k">if</span> <span class="pl-s1">closed_eye_consecutive</span> <span class="pl-c1">&gt;</span> <span class="pl-c1">MAX_CLOSED_FRAMES</span>:</div></div></div><div data-key="191" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3820px); min-height: auto;"><div><div id="LC192" class="react-file-line html-div" data-testid="code-cell" data-line-number="192" inert="inert" style="position: relative;">                        <span class="pl-s1">good_start_time</span> <span class="pl-c1">=</span> <span class="pl-c1">None</span></div></div></div><div data-key="192" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3840px); min-height: auto;"><div><div id="LC193" class="react-file-line html-div" data-testid="code-cell" data-line-number="193" inert="inert" style="position: relative;">            <span class="pl-k">else</span>:</div></div></div><div data-key="193" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3860px); min-height: auto;"><div><div id="LC194" class="react-file-line html-div" data-testid="code-cell" data-line-number="194" inert="inert" style="position: relative;">                <span class="pl-s1">good_start_time</span> <span class="pl-c1">=</span> <span class="pl-c1">None</span></div></div></div><div data-key="194" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3880px); min-height: auto;"><div><div id="LC195" class="react-file-line html-div" data-testid="code-cell" data-line-number="195" inert="inert" style="position: relative;">                <span class="pl-s1">closed_eye_consecutive</span> <span class="pl-c1">=</span> <span class="pl-c1">0</span></div></div></div><div data-key="195" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3900px); min-height: auto;"><div><div id="LC196" class="react-file-line html-div" data-testid="code-cell" data-line-number="196" inert="inert" style="position: relative;">
</div></div></div><div data-key="196" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3920px); min-height: auto;"><div><div id="LC197" class="react-file-line html-div" data-testid="code-cell" data-line-number="197" inert="inert" style="position: relative;">            <span class="pl-c"># If criteria are met for 10 seconds and attendance not yet marked.</span></div></div></div><div data-key="197" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3940px); min-height: auto;"><div><div id="LC198" class="react-file-line html-div" data-testid="code-cell" data-line-number="198" inert="inert" style="position: relative;">            <span class="pl-k">if</span> <span class="pl-s1">good_start_time</span> <span class="pl-c1"><span class="pl-c1">is</span> <span class="pl-c1">not</span></span> <span class="pl-c1">None</span>:</div></div></div><div data-key="198" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3960px); min-height: auto;"><div><div id="LC199" class="react-file-line html-div" data-testid="code-cell" data-line-number="199" inert="inert" style="position: relative;">                <span class="pl-s1">elapsed</span> <span class="pl-c1">=</span> <span class="pl-s1">current_time</span> <span class="pl-c1">-</span> <span class="pl-s1">good_start_time</span></div></div></div><div data-key="199" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(3980px); min-height: auto;"><div><div id="LC200" class="react-file-line html-div" data-testid="code-cell" data-line-number="200" inert="inert" style="position: relative;">                <span class="pl-s1">cv2</span>.<span class="pl-c1">putText</span>(<span class="pl-s1">frame</span>, <span class="pl-s">f"Good for <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">elapsed</span>:.1f<span class="pl-kos">}</span></span>s"</span>, (<span class="pl-s1">x</span>, <span class="pl-s1">y</span> <span class="pl-c1">-</span> <span class="pl-c1">40</span>),</div></div></div><div data-key="200" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4000px); min-height: auto;"><div><div id="LC201" class="react-file-line html-div" data-testid="code-cell" data-line-number="201" inert="inert" style="position: relative;">                            <span class="pl-s1">cv2</span>.<span class="pl-c1">FONT_HERSHEY_SIMPLEX</span>, <span class="pl-c1">FONT_SCALE</span>, (<span class="pl-c1">0</span>, <span class="pl-c1">255</span>, <span class="pl-c1">0</span>), <span class="pl-c1">FONT_THICKNESS</span>)</div></div></div><div data-key="201" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4020px); min-height: auto;"><div><div id="LC202" class="react-file-line html-div" data-testid="code-cell" data-line-number="202" inert="inert" style="position: relative;">                <span class="pl-k">if</span> <span class="pl-s1">elapsed</span> <span class="pl-c1">&gt;=</span> <span class="pl-c1">ATTENDANCE_DURATION</span> <span class="pl-c1">and</span> <span class="pl-c1">not</span> <span class="pl-s1">attendance_marked</span>:</div></div></div><div data-key="202" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4040px); min-height: auto;"><div><div id="LC203" class="react-file-line html-div" data-testid="code-cell" data-line-number="203" inert="inert" style="position: relative;">                    <span class="pl-s1">attendance_marked</span> <span class="pl-c1">=</span> <span class="pl-c1">True</span></div></div></div><div data-key="203" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4060px); min-height: auto;"><div><div id="LC204" class="react-file-line html-div" data-testid="code-cell" data-line-number="204" inert="inert" style="position: relative;">                    <span class="pl-s1">attendance_marked_time</span> <span class="pl-c1">=</span> <span class="pl-s1">current_time</span></div></div></div><div data-key="204" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4080px); min-height: auto;"><div><div id="LC205" class="react-file-line html-div" data-testid="code-cell" data-line-number="205" inert="inert" style="position: relative;">            <span class="pl-k">else</span>:</div></div></div><div data-key="205" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4100px); min-height: auto;"><div><div id="LC206" class="react-file-line html-div" data-testid="code-cell" data-line-number="206" inert="inert" style="position: relative;">                <span class="pl-s1">cv2</span>.<span class="pl-c1">putText</span>(<span class="pl-s1">frame</span>, <span class="pl-s">"Reset Timer"</span>, (<span class="pl-s1">x</span>, <span class="pl-s1">y</span> <span class="pl-c1">-</span> <span class="pl-c1">40</span>),</div></div></div><div data-key="206" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4120px); min-height: auto;"><div><div id="LC207" class="react-file-line html-div" data-testid="code-cell" data-line-number="207" inert="inert" style="position: relative;">                            <span class="pl-s1">cv2</span>.<span class="pl-c1">FONT_HERSHEY_SIMPLEX</span>, <span class="pl-c1">FONT_SCALE</span>, (<span class="pl-c1">0</span>, <span class="pl-c1">0</span>, <span class="pl-c1">255</span>), <span class="pl-c1">FONT_THICKNESS</span>)</div></div></div><div data-key="207" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4140px); min-height: auto;"><div><div id="LC208" class="react-file-line html-div" data-testid="code-cell" data-line-number="208" inert="inert" style="position: relative;">        <span class="pl-k">else</span>:</div></div></div><div data-key="208" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4160px); min-height: auto;"><div><div id="LC209" class="react-file-line html-div" data-testid="code-cell" data-line-number="209" inert="inert" style="position: relative;">            <span class="pl-s1">good_start_time</span> <span class="pl-c1">=</span> <span class="pl-c1">None</span></div></div></div><div data-key="209" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4180px); min-height: auto;"><div><div id="LC210" class="react-file-line html-div" data-testid="code-cell" data-line-number="210" inert="inert" style="position: relative;">            <span class="pl-s1">closed_eye_consecutive</span> <span class="pl-c1">=</span> <span class="pl-c1">0</span></div></div></div><div data-key="210" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4200px); min-height: auto;"><div><div id="LC211" class="react-file-line html-div" data-testid="code-cell" data-line-number="211" inert="inert" style="position: relative;">
</div></div></div><div data-key="211" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4220px); min-height: auto;"><div><div id="LC212" class="react-file-line html-div" data-testid="code-cell" data-line-number="212" inert="inert" style="position: relative;">        <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="212" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4240px); min-height: auto;"><div><div id="LC213" class="react-file-line html-div" data-testid="code-cell" data-line-number="213" inert="inert" style="position: relative;">        <span class="pl-c"># 7. Display the Attendance Marked message at the top-right (if within display duration)</span></div></div></div><div data-key="213" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4260px); min-height: auto;"><div><div id="LC214" class="react-file-line html-div" data-testid="code-cell" data-line-number="214" inert="inert" style="position: relative;">        <span class="pl-c"># -----------------------------</span></div></div></div><div data-key="214" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4280px); min-height: auto;"><div><div id="LC215" class="react-file-line html-div" data-testid="code-cell" data-line-number="215" inert="inert" style="position: relative;">        <span class="pl-k">if</span> <span class="pl-s1">attendance_marked_time</span> <span class="pl-c1"><span class="pl-c1">is</span> <span class="pl-c1">not</span></span> <span class="pl-c1">None</span>:</div></div></div><div data-key="215" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4300px); min-height: auto;"><div><div id="LC216" class="react-file-line html-div" data-testid="code-cell" data-line-number="216" inert="inert" style="position: relative;">            <span class="pl-k">if</span> <span class="pl-s1">current_time</span> <span class="pl-c1">-</span> <span class="pl-s1">attendance_marked_time</span> <span class="pl-c1">&lt;=</span> <span class="pl-c1">ATTENDANCE_MESSAGE_DURATION</span>:</div></div></div><div data-key="216" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4320px); min-height: auto;"><div><div id="LC217" class="react-file-line html-div" data-testid="code-cell" data-line-number="217" inert="inert" style="position: relative;">                <span class="pl-c"># Calculate position at top right</span></div></div></div><div data-key="217" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4340px); min-height: auto;"><div><div id="LC218" class="react-file-line html-div" data-testid="code-cell" data-line-number="218" inert="inert" style="position: relative;">                <span class="pl-s1">text</span> <span class="pl-c1">=</span> <span class="pl-s">"Attendance Marked"</span></div></div></div><div data-key="218" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4360px); min-height: auto;"><div><div id="LC219" class="react-file-line html-div" data-testid="code-cell" data-line-number="219" inert="inert" style="position: relative;">                (<span class="pl-s1">text_width</span>, <span class="pl-s1">text_height</span>), <span class="pl-s1">_</span> <span class="pl-c1">=</span> <span class="pl-s1">cv2</span>.<span class="pl-c1">getTextSize</span>(<span class="pl-s1">text</span>, <span class="pl-s1">cv2</span>.<span class="pl-c1">FONT_HERSHEY_SIMPLEX</span>, <span class="pl-c1">ATTENDANCE_FONT_SCALE</span>, <span class="pl-c1">FONT_THICKNESS</span>)</div></div></div><div data-key="219" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4380px); min-height: auto;"><div><div id="LC220" class="react-file-line html-div" data-testid="code-cell" data-line-number="220" inert="inert" style="position: relative;">                <span class="pl-c"># Place text with some margin from top-right corner</span></div></div></div><div data-key="220" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4400px); min-height: auto;"><div><div id="LC221" class="react-file-line html-div" data-testid="code-cell" data-line-number="221" inert="inert" style="position: relative;">                <span class="pl-s1">pos</span> <span class="pl-c1">=</span> (<span class="pl-s1">frame</span>.<span class="pl-c1">shape</span>[<span class="pl-c1">1</span>] <span class="pl-c1">-</span> <span class="pl-s1">text_width</span> <span class="pl-c1">-</span> <span class="pl-c1">20</span>, <span class="pl-s1">text_height</span> <span class="pl-c1">+</span> <span class="pl-c1">20</span>)</div></div></div><div data-key="221" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4420px); min-height: auto;"><div><div id="LC222" class="react-file-line html-div" data-testid="code-cell" data-line-number="222" inert="inert" style="position: relative;">                <span class="pl-s1">cv2</span>.<span class="pl-c1">putText</span>(<span class="pl-s1">frame</span>, <span class="pl-s1">text</span>, <span class="pl-s1">pos</span>, <span class="pl-s1">cv2</span>.<span class="pl-c1">FONT_HERSHEY_SIMPLEX</span>, <span class="pl-c1">ATTENDANCE_FONT_SCALE</span>, (<span class="pl-c1">0</span>, <span class="pl-c1">255</span>, <span class="pl-c1">0</span>), <span class="pl-c1">FONT_THICKNESS</span>)</div></div></div><div data-key="222" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4440px); min-height: auto;"><div><div id="LC223" class="react-file-line html-div" data-testid="code-cell" data-line-number="223" inert="inert" style="position: relative;">            <span class="pl-k">else</span>:</div></div></div><div data-key="223" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4460px); min-height: auto;"><div><div id="LC224" class="react-file-line html-div" data-testid="code-cell" data-line-number="224" inert="inert" style="position: relative;">                <span class="pl-c"># After 2 seconds, clear the attendance display time.</span></div></div></div><div data-key="224" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4480px); min-height: auto;"><div><div id="LC225" class="react-file-line html-div" data-testid="code-cell" data-line-number="225" inert="inert" style="position: relative;">                <span class="pl-s1">attendance_marked_time</span> <span class="pl-c1">=</span> <span class="pl-c1">None</span></div></div></div><div data-key="225" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4500px); min-height: auto;"><div><div id="LC226" class="react-file-line html-div" data-testid="code-cell" data-line-number="226" inert="inert" style="position: relative;">
</div></div></div><div data-key="226" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4520px); min-height: auto;"><div><div id="LC227" class="react-file-line html-div" data-testid="code-cell" data-line-number="227" inert="inert" style="position: relative;">        <span class="pl-s1">cv2</span>.<span class="pl-c1">imshow</span>(<span class="pl-s">"Face &amp; Eye Detection with Attendance"</span>, <span class="pl-s1">frame</span>)</div></div></div><div data-key="227" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4540px); min-height: auto;"><div><div id="LC228" class="react-file-line html-div" data-testid="code-cell" data-line-number="228" inert="inert" style="position: relative;">        <span class="pl-k">if</span> <span class="pl-s1">cv2</span>.<span class="pl-c1">waitKey</span>(<span class="pl-c1">1</span>) <span class="pl-c1">&amp;</span> <span class="pl-c1">0xFF</span> <span class="pl-c1">==</span> <span class="pl-en">ord</span>(<span class="pl-s">'q'</span>):</div></div></div><div data-key="228" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4560px); min-height: auto;"><div><div id="LC229" class="react-file-line html-div" data-testid="code-cell" data-line-number="229" inert="inert" style="position: relative;">            <span class="pl-k">break</span></div></div></div><div data-key="229" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4580px); min-height: auto;"><div><div id="LC230" class="react-file-line html-div" data-testid="code-cell" data-line-number="230" inert="inert" style="position: relative;">
</div></div></div><div data-key="230" class="child-of-line-42  react-code-text react-code-line-contents virtual" style="transform: translateY(4600px); min-height: auto;"><div><div id="LC231" class="react-file-line html-div" data-testid="code-cell" data-line-number="231" inert="inert" style="position: relative;">    <span class="pl-s1">cap</span>.<span class="pl-c1">release</span>()</div></div></div><div data-key="231" class="react-code-text react-code-line-contents virtual" style="transform: translateY(4620px); min-height: auto;"><div><div id="LC232" class="react-file-line html-div" data-testid="code-cell" data-line-number="232" inert="inert" style="position: relative;">    <span class="pl-s1">cv2</span>.<span class="pl-c1">destroyAllWindows</span>()</div></div></div><div data-key="232" class="react-code-text react-code-line-contents virtual" style="transform: translateY(4640px); min-height: auto;"><div><div id="LC233" class="react-file-line html-div" data-testid="code-cell" data-line-number="233" inert="inert" style="position: relative;">
</div></div></div><div data-key="233" class="react-code-text react-code-line-contents virtual" style="transform: translateY(4660px); min-height: auto;"><div><div id="LC234" class="react-file-line html-div" data-testid="code-cell" data-line-number="234" inert="inert" style="position: relative;"><span class="pl-c"># -----------------------------</span></div></div></div><div data-key="234" class="react-code-text react-code-line-contents virtual" style="transform: translateY(4680px); min-height: auto;"><div><div id="LC235" class="react-file-line html-div" data-testid="code-cell" data-line-number="235" inert="inert" style="position: relative;"><span class="pl-c"># Main execution</span></div></div></div><div data-key="235" class="react-code-text react-code-line-contents virtual" style="transform: translateY(4700px); min-height: auto;"><div><div id="LC236" class="react-file-line html-div" data-testid="code-cell" data-line-number="236" inert="inert" style="position: relative;"><span class="pl-c"># -----------------------------</span></div></div></div><div data-key="236" class="react-code-text react-code-line-contents virtual" style="transform: translateY(4720px); min-height: auto;"><div><div id="LC237" class="react-file-line html-div" data-testid="code-cell" data-line-number="237" inert="inert" style="position: relative;"><span class="pl-k">if</span> <span class="pl-s1">__name__</span> <span class="pl-c1">==</span> <span class="pl-s">"__main__"</span>:</div></div></div><div data-key="237" class="react-code-text react-code-line-contents virtual" style="transform: translateY(4740px); min-height: auto;"><div><div id="LC238" class="react-file-line html-div" data-testid="code-cell" data-line-number="238" inert="inert" style="position: relative;">    <span class="pl-c"># Pass a video file path or 0 for webcam.</span></div></div></div><div data-key="238" class="react-code-text react-code-line-contents virtual" style="transform: translateY(4760px); min-height: auto;"><div><div id="LC239" class="react-file-line html-div" data-testid="code-cell" data-line-number="239" inert="inert" style="position: relative;">    <span class="pl-en">detect_face_eyes</span>(<span class="pl-c1">0</span>)</div></div></div></div><button hidden="" data-hotkey="Control+a"></button></div></div><div class="Box-sc-g0xbh4-0 fXFeWj"><div class="Box-sc-g0xbh4-0 dOCddK"></div></div></div></div><div id="copilot-button-container"></div></div><div id="highlighted-line-menu-container"></div></div></div></section></div></div></div> <!-- --> <!-- --> </div></div></div><div class="Box-sc-g0xbh4-0"></div></div></div></div></div><div id="find-result-marks-container" class="Box-sc-g0xbh4-0 cCoXib"></div><button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="Control+F6,Control+Shift+F6"></button><button hidden="" data-hotkey="Control+F6,Control+Shift+F6"></button></div> <!-- --> <!-- --> <script type="application/json" id="__PRIMER_DATA_:R0:__">{"resolvedServerColorMode":"day"}</script></div>
</react-app>




  </div>

</turbo-frame>

    </main>
  </div>

  </div>

          <footer class="footer pt-8 pb-6 f6 color-fg-muted p-responsive" role="contentinfo" hidden="">
  <h2 class="sr-only">Footer</h2>

  


  <div class="d-flex flex-justify-center flex-items-center flex-column-reverse flex-lg-row flex-wrap flex-lg-nowrap">
    <div class="d-flex flex-items-center flex-shrink-0 mx-2">
      <a aria-label="Homepage" title="GitHub" class="footer-octicon mr-2" href="https://github.com/">
        <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-mark-github">
    <path d="M12 1C5.9225 1 1 5.9225 1 12C1 16.8675 4.14875 20.9787 8.52125 22.4362C9.07125 22.5325 9.2775 22.2025 9.2775 21.9137C9.2775 21.6525 9.26375 20.7862 9.26375 19.865C6.5 20.3737 5.785 19.1912 5.565 18.5725C5.44125 18.2562 4.905 17.28 4.4375 17.0187C4.0525 16.8125 3.5025 16.3037 4.42375 16.29C5.29 16.2762 5.90875 17.0875 6.115 17.4175C7.105 19.0812 8.68625 18.6137 9.31875 18.325C9.415 17.61 9.70375 17.1287 10.02 16.8537C7.5725 16.5787 5.015 15.63 5.015 11.4225C5.015 10.2262 5.44125 9.23625 6.1425 8.46625C6.0325 8.19125 5.6475 7.06375 6.2525 5.55125C6.2525 5.55125 7.17375 5.2625 9.2775 6.67875C10.1575 6.43125 11.0925 6.3075 12.0275 6.3075C12.9625 6.3075 13.8975 6.43125 14.7775 6.67875C16.8813 5.24875 17.8025 5.55125 17.8025 5.55125C18.4075 7.06375 18.0225 8.19125 17.9125 8.46625C18.6138 9.23625 19.04 10.2125 19.04 11.4225C19.04 15.6437 16.4688 16.5787 14.0213 16.8537C14.42 17.1975 14.7638 17.8575 14.7638 18.8887C14.7638 20.36 14.75 21.5425 14.75 21.9137C14.75 22.2025 14.9563 22.5462 15.5063 22.4362C19.8513 20.9787 23 16.8537 23 12C23 5.9225 18.0775 1 12 1Z"></path>
</svg>
</a>
      <span>
        © 2025 GitHub,&nbsp;Inc.
      </span>
    </div>

    <nav aria-label="Footer">
      <h3 class="sr-only" id="sr-footer-heading">Footer navigation</h3>

      <ul class="list-style-none d-flex flex-justify-center flex-wrap mb-2 mb-lg-0" aria-labelledby="sr-footer-heading">

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to Terms&quot;,&quot;label&quot;:&quot;text:terms&quot;}" href="https://docs.github.com/site-policy/github-terms/github-terms-of-service" data-view-component="true" class="Link--secondary Link">Terms</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to privacy&quot;,&quot;label&quot;:&quot;text:privacy&quot;}" href="https://docs.github.com/site-policy/privacy-policies/github-privacy-statement" data-view-component="true" class="Link--secondary Link">Privacy</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to security&quot;,&quot;label&quot;:&quot;text:security&quot;}" href="https://github.com/security" data-view-component="true" class="Link--secondary Link">Security</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to status&quot;,&quot;label&quot;:&quot;text:status&quot;}" href="https://www.githubstatus.com/" data-view-component="true" class="Link--secondary Link">Status</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to docs&quot;,&quot;label&quot;:&quot;text:docs&quot;}" href="https://docs.github.com/" data-view-component="true" class="Link--secondary Link">Docs</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to contact&quot;,&quot;label&quot;:&quot;text:contact&quot;}" href="https://support.github.com/?tags=dotcom-footer" data-view-component="true" class="Link--secondary Link">Contact</a>
          </li>

          <li class="mx-2">
  <cookie-consent-link data-catalyst="">
    <button type="button" class="Link--secondary underline-on-hover border-0 p-0 color-bg-transparent" data-action="click:cookie-consent-link#showConsentManagement" data-analytics-event="{&quot;location&quot;:&quot;footer&quot;,&quot;action&quot;:&quot;cookies&quot;,&quot;context&quot;:&quot;subfooter&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;cookies_link_subfooter_footer&quot;}">
      Manage cookies
    </button>
  </cookie-consent-link>
</li>

<li class="mx-2">
  <cookie-consent-link data-catalyst="">
    <button type="button" class="Link--secondary underline-on-hover border-0 p-0 color-bg-transparent" data-action="click:cookie-consent-link#showConsentManagement" data-analytics-event="{&quot;location&quot;:&quot;footer&quot;,&quot;action&quot;:&quot;dont_share_info&quot;,&quot;context&quot;:&quot;subfooter&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;dont_share_info_link_subfooter_footer&quot;}">
      Do not share my personal information
    </button>
  </cookie-consent-link>
</li>

      </ul>
    </nav>
  </div>
</footer>



    <ghcc-consent id="ghcc" class="position-fixed bottom-0 left-0" style="z-index: 999999" data-initial-cookie-consent-allowed="" data-cookie-consent-required="false" data-catalyst=""></ghcc-consent>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error" hidden="">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-alert">
    <path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path>
</svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"></path>
</svg>
    </button>
    You can’t perform that action at this time.
  </div>

    <template id="site-details-dialog"></template>

    <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box color-shadow-large" style="width:360px;"></div>
</div>

    <template id="snippet-clipboard-copy-button"></template>
<template id="snippet-clipboard-copy-button-unpositioned"></template>




    </div>

    <div id="js-global-screen-reader-notice" class="sr-only mt-n1" aria-live="polite" aria-atomic="true">Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection/App.py at main · Rafael-ZP/Lock-In--A_Secure_Attendance_via_Gaze_and_Blink_Detection · GitHub</div>
    <div id="js-global-screen-reader-notice-assertive" class="sr-only mt-n1" aria-live="assertive" aria-atomic="true"></div>
  


<div class="sr-only mt-n1" id="screenReaderAnnouncementDiv" role="alert" data-testid="screenReaderAnnouncement" aria-live="assertive">&nbsp;</div></body></html>