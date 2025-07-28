document.addEventListener("DOMContentLoaded", function () {
  const toggleBtn = document.querySelector("[data-nav-toggle-btn]");
  const mobileNav = document.querySelector(".mobile-nav");
  const menuIcon = toggleBtn.querySelector(".menu-icon");
  const closeIcon = toggleBtn.querySelector(".close-icon");

  // Event listener to toggle menu visibility
  toggleBtn.addEventListener("click", function () {
    // Toggle mobile navigation
    mobileNav.classList.toggle("d-none");

    // Toggle menu and close icons
    menuIcon.classList.toggle("d-none");
    closeIcon.classList.toggle("d-none");
  });
});


swiper1 = new Swiper(".swiper-container", {
  slidesPerView: 1,
  spaceBetween: 30,
  loop: true,
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },

  // And if we need scrollbar
  scrollbar: {
    el: '.swiper-scrollbar',
  },
  navigation: {
    nextEl: ".swiper-button-next",
    prevEl: ".swiper-button-prev",
  },
});