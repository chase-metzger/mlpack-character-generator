#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QTableWidget>

#include <mlpack/core.hpp>
#include <mlpack/core/arma_extend/arma_extend.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
		Q_OBJECT

	public:
		MainWindow(QWidget *parent = nullptr);
		~MainWindow();

		void setResultText(const QString &text);
		void setDataset(const arma::mat &data);

	private:
		Ui::MainWindow *ui;
	//	QLabel *resultLabel = nullptr;
	//	QTableWidget *datasetTable = nullptr;
};
#endif // MAINWINDOW_H
