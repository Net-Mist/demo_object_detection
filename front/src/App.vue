<template lang="pug">
  v-app
    v-navigation-drawer(v-model="gui.nav_drawer_visible" clipped app)
      v-list-item
        v-list-item-content
          v-list-item-title
            v-switch(v-model="gui.fullscreen_image" label="Fullscreen" primary)

    v-app-bar(app clipped-left)
      v-app-bar-nav-icon(@click.stop="gui.nav_drawer_visible = !gui.nav_drawer_visible")
      v-toolbar-title Object detection

    v-content
      v-container(fluid) 
        v-layout(justify-space-around align-space-around)
          v-flex(xs5 v-if="!gui.fullscreen_image")
            v-img(src="/video_feed" aspect-ratio="1.7")
          v-flex(:xs5="!gui.fullscreen_image" :xs10="gui.fullscreen_image") 
            v-img(src="/inference_feed" aspect-ratio="1.7")
    v-footer(app) 
      span(class="px-3") &copy; Sébastien IOOSS {{ new Date().getFullYear() }}
</template>

<script>

export default {
  name: "App",
  data() {
    return {
      gui: {
        nav_drawer_visible: false,
        fullscreen_image: false
      },
    };
  },
  mounted: function() {
  }
};
</script>
