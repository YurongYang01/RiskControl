from flask import Blueprint, render_template, redirect, url_for

views_bp = Blueprint('risk_cot_views', __name__)

@views_bp.route('/')
def index():
    return redirect(url_for('risk_cot_views.inference'))

@views_bp.route('/prompt_design')
def prompt_design():
    return render_template('risk_cot/prompt_design.html', active_page='prompt_design')

@views_bp.route('/inference')
def inference():
    return render_template('risk_cot/inference.html', active_page='cot_synthesis')

@views_bp.route('/distillation')
def distillation():
    return render_template('risk_cot/distillation.html', active_page='model_distillation')
