
  var scrollElement = '#body';
  var $scrollElement; 
    var initScrollLeft; 

  /* Smooth scrolling of links between panels */
  $(function() {
    var $panels = $('.panel');

    $scrollElement = $(scrollElement);
    $panels.each(function(a) {
      var $panel = $(this);
      var hash = '#' + this.id;
$panel.css({left: a*$panel.width()});
      $('a[href="' + hash + '"]').click(function(event) {$(".nav a").css("color","white");$(this).css("color","green").css("font-weight","bolder");
        $scrollElement.stop().animate({
        }, 500, 'swing', function() {
          window.location.hash = hash;
        });

        event.preventDefault();
      });
    });
  });
   