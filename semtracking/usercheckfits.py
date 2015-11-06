import pandas
import plot


class UserCheckFits:
    # Store fits so that we can use it in event callback
    fits_for_user_check = pandas.DataFrame()

    def __init__(self, filename, micron_per_pixel):
        """
        Let user manually check fits, removing them by clicking
        :return:
        """
        self.fits_for_user_check = pandas.DataFrame.from_csv(filename)

        # Set scale in pixels
        self.fits_for_user_check /= micron_per_pixel

        self.fits_for_user_check['remove'] = False

    def user_check_fits(self, image):
        plot.plot_fits_for_user_confirmation(self.fits_for_user_check, image, self.on_pick)

        mask = (self.fits_for_user_check['remove'] == False)
        fits_for_user_check = self.fits_for_user_check[mask]

        fits_for_user_check.drop('remove', axis=1, inplace=True)

        # Update indices
        self.fits_for_user_check.index = range(1, len(fits_for_user_check) + 1)

        return self.fits_for_user_check

    @classmethod
    def on_pick(cls, event):
        """
        User clicked on a fit
        :param event:
        """
        # Get index from label
        fit_number = int(event.artist.get_label())

        if not cls.fits_for_user_check['remove'][fit_number]:
            event.artist.set_edgecolor('r')
            plot.set_annotation_color(fit_number, 'r')
            cls.fits_for_user_check['remove'][fit_number] = True
        else:
            event.artist.set_edgecolor('b')
            plot.set_annotation_color(fit_number, 'b')
            cls.fits_for_user_check['remove'][fit_number] = False

        event.canvas.draw()
