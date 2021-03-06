/*
Copyright (c) 2012, Daniel Moreno and Gabriel Taubin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Brown University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL DANIEL MORENO AND GABRIEL TAUBIN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __CALIBRATIONDIALOG_HPP__
#define __CALIBRATIONDIALOG_HPP__

#include <QDialog>

#include "ui_CalibrationDialog.h"

class CalibrationDialog : public QDialog, public Ui::CalibrationDialog
{
    Q_OBJECT

public:
    CalibrationDialog(QWidget * parent = 0, Qt::WindowFlags flags = 0);
    ~CalibrationDialog();

    inline void set_current_message(const QString & text) {current_message_label->setText(text);}
    void reset(void);
    inline void set_progress_total(unsigned value) {_total = value; progress_bar->setMaximum(_total);}
    inline void set_progress_value(unsigned value) {progress_bar->setValue(value);}

    void finish(void);

    void message(const QString & text);

    inline bool canceled(void) const {return _cancel;}

public slots:
    void on_close_cancel_button_clicked(bool checked = false);

private:
    unsigned _total;
    bool _cancel;
};


#endif  /* __CALIBRATIONDIALOG_HPP__ */
